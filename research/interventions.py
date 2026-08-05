"""Run post-training causal ablations on recurrent checkpoint channels."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark.batches import prepare_batch
from benchmark.runner import _autocast
from data import make_dataloaders
from research.geometry import _load_trained_model


@dataclass(frozen=True)
class PredictionMeasurement:
    metrics: dict[str, float | int]
    predictions: tuple[tuple[int, ...], ...]
    correct: tuple[bool, ...]


def _measure_predictions(
    model: nn.Module,
    dataloader,
    manifest,
    device: torch.device,
) -> PredictionMeasurement:
    total_loss = 0.0
    token_count = 0
    predictions: list[tuple[int, ...]] = []
    correct: list[bool] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets, attention_mask, target_positions = prepare_batch(
                batch,
                device,
            )
            if target_positions is None:
                raise ValueError("intervention analysis requires separate output positions")
            with _autocast(manifest, device):
                logits, _ = model(input_ids, attention_mask=attention_mask)
            batch_indices = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[
                batch_indices,
                target_positions.clamp_min(0),
            ].float()
            valid = targets != -100
            total_loss += float(
                F.cross_entropy(
                    token_logits[valid],
                    targets[valid],
                    reduction="sum",
                ).item()
            )
            token_count += int(valid.sum().item())
            token_predictions = token_logits.argmax(dim=-1)
            for row in range(targets.shape[0]):
                row_valid = valid[row]
                if not row_valid.any().item():
                    continue
                predicted = tuple(
                    int(value)
                    for value in token_predictions[row][row_valid].detach().cpu()
                )
                expected = tuple(
                    int(value) for value in targets[row][row_valid].detach().cpu()
                )
                predictions.append(predicted)
                correct.append(predicted == expected)

    if not predictions or token_count == 0:
        raise ValueError("intervention split contains no target examples")
    correct_count = sum(correct)
    return PredictionMeasurement(
        metrics={
            "loss": total_loss / token_count,
            "exact_accuracy": correct_count / len(correct),
            "correct_examples": correct_count,
            "example_count": len(correct),
            "target_token_count": token_count,
        },
        predictions=tuple(predictions),
        correct=tuple(correct),
    )


def _compare_measurement(
    measurement: PredictionMeasurement,
    baseline: PredictionMeasurement,
) -> dict[str, float | int]:
    if len(measurement.predictions) != len(baseline.predictions):
        raise ValueError("intervention and baseline example counts differ")

    matching_examples = 0
    changed_tokens = 0
    compared_tokens = 0
    correctness_flips = 0
    for prediction, reference, correct, reference_correct in zip(
        measurement.predictions,
        baseline.predictions,
        measurement.correct,
        baseline.correct,
        strict=True,
    ):
        if len(prediction) != len(reference):
            raise ValueError("intervention and baseline target lengths differ")
        matching_examples += prediction == reference
        changed_tokens += sum(
            candidate != expected
            for candidate, expected in zip(prediction, reference, strict=True)
        )
        compared_tokens += len(reference)
        correctness_flips += correct != reference_correct

    example_count = len(baseline.predictions)
    return {
        **measurement.metrics,
        "accuracy_delta": (
            float(measurement.metrics["exact_accuracy"])
            - float(baseline.metrics["exact_accuracy"])
        ),
        "loss_delta": (
            float(measurement.metrics["loss"])
            - float(baseline.metrics["loss"])
        ),
        "prediction_agreement": matching_examples / example_count,
        "prediction_change_rate": 1 - matching_examples / example_count,
        "token_change_rate": changed_tokens / compared_tokens,
        "correctness_flip_rate": correctness_flips / example_count,
    }


@contextmanager
def _output_hook(
    module: nn.Module,
    transform: Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor],
) -> Iterator[None]:
    handle = module.register_forward_hook(transform)
    try:
        yield
    finally:
        handle.remove()


def _zero_output(module: nn.Module):
    return _output_hook(
        module,
        lambda _module, _inputs, output: torch.zeros_like(output),
    )


def _freeze_output(module: nn.Module):
    return _output_hook(
        module,
        lambda _module, inputs, _output: inputs[1],
    )


def _freeze_after_step(model: nn.Module, completed_steps: int):
    execution_loops = (
        model.architecture.training_loop_cap
        if model.architecture.training_loop_cap is not None
        else model.architecture.num_loops
    )
    call_index = 0

    def transform(_module, inputs, output):
        nonlocal call_index
        step_index = call_index % execution_loops
        call_index += 1
        return output if step_index < completed_steps else inputs[1]

    return _output_hook(model.step.residue_cell, transform)


def _swap_output(module: nn.Module, tokens_per_example: int):
    def transform(_module, _inputs, output):
        width = output.shape[-1]
        states = output.reshape(-1, tokens_per_example, width)
        return states.roll(shifts=1, dims=0).reshape_as(output)

    return _output_hook(module, transform)


@contextmanager
def _replace_initial_state(
    model: nn.Module,
    state_index: int,
    transform: Callable[[Tensor], Tensor],
) -> Iterator[None]:
    original = model._initial_states

    def replaced(*args, **kwargs):
        states = list(original(*args, **kwargs))
        states[state_index] = transform(states[state_index])
        return tuple(states)

    model._initial_states = replaced
    try:
        yield
    finally:
        model._initial_states = original


def _causal_state_interventions(model: nn.Module):
    residue_cell = model.step.residue_cell
    scratch_cell = model.step.scratch_cell
    if scratch_cell is None:
        raise ValueError("causal-state intervention gate requires scratch state")
    digit_slots = int(model.architecture.digit_slots)
    scratch_tokens = int(model.architecture.scratch_tokens)
    return {
        "baseline": nullcontext,
        "freeze_residue_all": lambda: _freeze_output(residue_cell),
        "freeze_residue_after_step_1": lambda: _freeze_after_step(model, 1),
        "freeze_residue_after_step_2": lambda: _freeze_after_step(model, 2),
        "zero_residue_transition": lambda: _zero_output(residue_cell),
        "swap_residue_transition": lambda: _swap_output(
            residue_cell,
            digit_slots,
        ),
        "zero_scratch_transition": lambda: _zero_output(scratch_cell),
        "swap_scratch_transition": lambda: _swap_output(
            scratch_cell,
            scratch_tokens,
        ),
        "zero_static_modulus": lambda: _replace_initial_state(
            model,
            0,
            torch.zeros_like,
        ),
        "swap_static_modulus": lambda: _replace_initial_state(
            model,
            0,
            lambda state: state.roll(shifts=1, dims=0),
        ),
    }


def _register_interventions(model: nn.Module):
    @contextmanager
    def zero_structured() -> Iterator[None]:
        modules = [
            module
            for module in (model.field_embedding, model.place_embedding)
            if module is not None
        ]
        handles = [
            module.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(output)
            )
            for module in modules
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    return {
        "baseline": nullcontext,
        "zero_feedback_state": lambda: _zero_output(model.state_projection),
        "zero_field_and_place": zero_structured,
        "zero_position": lambda: _zero_output(model.position_embedding),
    }


def analyze_interventions(
    *,
    run_dir: Path,
    split: str,
    output_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest, checkpoint, model = _load_trained_model(run_dir, device)
    seed = int(checkpoint["seed"])
    data_config = replace(
        manifest.data,
        seed=seed,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dataloaders = make_dataloaders(data_config, device=device)
    if split not in dataloaders:
        raise ValueError(f"unknown split {split!r}; available={sorted(dataloaders)}")
    dataloader = dataloaders[split]

    if hasattr(model, "decode_residue"):
        factories = _causal_state_interventions(model)
        architecture = "causal_state"
    else:
        factories = _register_interventions(model)
        architecture = "register"

    measurements: dict[str, PredictionMeasurement] = {}
    for name, factory in factories.items():
        with factory():
            measurements[name] = _measure_predictions(
                model,
                dataloader,
                manifest,
                device,
            )
    measurements["baseline_repeat"] = _measure_predictions(
        model,
        dataloader,
        manifest,
        device,
    )

    baseline = measurements["baseline"]
    result = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "manifest": manifest.name,
        "seed": seed,
        "split": split,
        "architecture": architecture,
        "measurements": {
            name: _compare_measurement(measurement, baseline)
            for name, measurement in measurements.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"INTERVENTION_RESULT={output_path}", flush=True)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze_interventions(
        run_dir=args.run_dir,
        split=args.split,
        output_path=args.output,
    )


if __name__ == "__main__":
    cli()
