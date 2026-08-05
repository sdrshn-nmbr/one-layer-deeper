"""Analyze recurrent readouts and representation geometry from trained checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from benchmark.runner import _load_submission_file, _make_model_spec
from benchmark.manifest import load_manifest
from benchmark.batches import prepare_batch
from data import make_dataloaders
from data.squaring_mod import DIGIT_OFFSET, TOKEN_IDS, number_tokens


def _center(matrix: Tensor) -> Tensor:
    return matrix.float() - matrix.float().mean(dim=0, keepdim=True)


def linear_cka(left: Tensor, right: Tensor) -> float:
    """Return feature-space linear CKA for two row-aligned matrices."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be row-aligned matrices")
    left = _center(left)
    right = _center(right)
    cross = left.T @ right
    left_gram = left.T @ left
    right_gram = right.T @ right
    denominator = left_gram.square().sum().sqrt() * right_gram.square().sum().sqrt()
    if denominator.item() == 0:
        raise ValueError("CKA is undefined for a constant representation")
    return float((cross.square().sum() / denominator).item())


def orthogonal_procrustes_similarity(left: Tensor, right: Tensor) -> float:
    """Return normalized similarity after the best orthogonal feature rotation."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("Procrustes inputs must have the same matrix shape")
    left = _center(left)
    right = _center(right)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm.item() == 0 or right_norm.item() == 0:
        raise ValueError("Procrustes similarity is undefined for a constant representation")
    singular_values = torch.linalg.svdvals((left / left_norm).T @ (right / right_norm))
    return float(singular_values.sum().clamp(max=1).item())


def _flatten_valid(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    return hidden[attention_mask.to(device=hidden.device, dtype=torch.bool)]


def _readout_metrics(
    model,
    hidden: Tensor,
    targets: Tensor,
    target_positions: Tensor,
) -> dict[str, float | int]:
    logits = model.head(model.final_norm(hidden)).float()
    return _logit_metrics(logits, targets, target_positions)


def _logit_metrics(
    logits: Tensor,
    targets: Tensor,
    target_positions: Tensor,
) -> dict[str, float | int]:
    valid = targets != -100
    batch_indices = torch.arange(logits.shape[0], device=logits.device)[:, None]
    token_logits = logits[batch_indices, target_positions.clamp_min(0)]
    predictions = token_logits.argmax(dim=-1)
    rows = valid.any(dim=1)
    exact = ((predictions == targets) | ~valid).all(dim=1)[rows]
    return {
        "exact_accuracy": float(exact.float().mean().item()),
        "correct_examples": int(exact.sum().item()),
        "example_count": int(exact.numel()),
        "token_cross_entropy": float(
            F.cross_entropy(token_logits[valid], targets[valid]).item()
        ),
    }


def _residue_readout_metrics(
    model,
    residue: Tensor,
    targets: Tensor,
    target_positions: Tensor,
    attention_mask: Tensor,
) -> dict[str, float | int]:
    logits = model.decode_residue(
        residue,
        attention_mask,
        attention_mask.shape[1],
    ).float()
    return _logit_metrics(logits, targets, target_positions)


def _transition_metrics(previous: Tensor, current: Tensor) -> dict[str, float | None]:
    try:
        cka = linear_cka(previous, current)
    except ValueError:
        cka = None
    try:
        procrustes = orthogonal_procrustes_similarity(previous, current)
    except ValueError:
        procrustes = None
    return {
        "linear_cka": cka,
        "procrustes_similarity": procrustes,
        "relative_l2_change": float(
            (
                torch.linalg.vector_norm(current - previous)
                / torch.linalg.vector_norm(previous).clamp_min(1e-12)
            ).item()
        ),
    }


def _state_statistics(states: tuple[Tensor, ...]) -> list[dict[str, float | int]]:
    statistics = []
    for step, state in enumerate(states):
        flattened = state.float().reshape(state.shape[0], -1)
        mean = flattened.mean(dim=0, keepdim=True)
        centered = flattened - mean
        total_energy = flattened.square().mean()
        shared_energy = mean.square().mean()
        centered_energy = centered.square().mean()
        singular_values = torch.linalg.svdvals(centered)
        spectral_energy = singular_values.square()
        if spectral_energy.sum().item() == 0:
            effective_rank = 0.0
        else:
            probabilities = spectral_energy / spectral_energy.sum()
            effective_rank = float(
                torch.exp(
                    -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                ).item()
            )
        statistics.append(
            {
                "step": step,
                "examples": flattened.shape[0],
                "features_per_example": flattened.shape[1],
                "total_rms": float(total_energy.sqrt().item()),
                "shared_rms": float(shared_energy.sqrt().item()),
                "centered_rms": float(centered_energy.sqrt().item()),
                "centered_energy_fraction": float(
                    (centered_energy / total_energy.clamp_min(1e-12)).item()
                ),
                "effective_rank": effective_rank,
            }
        )
    return statistics


def _pair_routing_statistics(model) -> dict[str, Any] | None:
    interaction = getattr(model.step, "pair_interaction", None)
    if interaction is None:
        return None
    weights = interaction.routing_weights().detach().float().cpu()
    slot_count = weights.shape[0]
    flattened = weights.flatten(1)
    entropy = -(flattened * flattened.clamp_min(1e-12).log()).sum(dim=-1)
    rows = []
    source_left = torch.arange(slot_count)[:, None]
    source_right = torch.arange(slot_count)[None, :]
    for output_slot in range(slot_count):
        top_index = int(flattened[output_slot].argmax().item())
        left_slot, right_slot = divmod(top_index, slot_count)
        convolution_mask = source_left + source_right == output_slot
        rows.append(
            {
                "output_slot": output_slot,
                "normalized_entropy": float(
                    (entropy[output_slot] / math.log(slot_count**2)).item()
                ),
                "effective_pair_count": float(entropy[output_slot].exp().item()),
                "top_left_slot": left_slot,
                "top_right_slot": right_slot,
                "top_weight": float(flattened[output_slot, top_index].item()),
                "sum_route_mass": float(
                    weights[output_slot][convolution_mask].sum().item()
                ),
            }
        )
    return {
        "route_logit_std": float(interaction.route_logits.detach().float().std().item()),
        "mean_symmetry_error": float(
            (weights - weights.transpose(1, 2)).abs().mean().item()
        ),
        "outputs": rows,
    }


def _gradient_sensitivity(model, batch, device: torch.device) -> dict[str, Any]:
    input_ids, targets, attention_mask, target_positions = prepare_batch(batch, device)
    if target_positions is None:
        raise ValueError("gradient sensitivity requires separate output positions")
    scratch_cell_outputs: list[Tensor] = []
    scratch_cell = getattr(model.step, "scratch_cell", None)
    handle = (
        scratch_cell.register_forward_hook(
            lambda _module, _inputs, output: scratch_cell_outputs.append(output)
        )
        if scratch_cell is not None
        else None
    )
    with torch.enable_grad():
        try:
            trace = model.forward_with_trace(input_ids, attention_mask=attention_mask)
        finally:
            if handle is not None:
                handle.remove()
        batch_indices = torch.arange(trace.logits.shape[0], device=device)[:, None]
        token_logits = trace.logits[
            batch_indices,
            target_positions.clamp_min(0),
        ].float()
        valid = targets != -100
        loss = F.cross_entropy(token_logits[valid], targets[valid])
        uses_scratch_state = bool(trace.scratch_states) and bool(
            trace.scratch_states[0].numel()
        )
        if uses_scratch_state and scratch_cell is not None:
            scratch_sources = (trace.scratch_states[0], *scratch_cell_outputs)
        elif uses_scratch_state:
            scratch_sources = trace.scratch_states
        else:
            scratch_sources = ()
        states = (*trace.residue_states, *scratch_sources)
        gradients = torch.autograd.grad(
            loss,
            states,
            allow_unused=True,
        )

    def summarize(channel_states, channel_gradients):
        rows = []
        for step, (state, gradient) in enumerate(
            zip(channel_states, channel_gradients, strict=True)
        ):
            if gradient is None:
                rows.append(
                    {
                        "step": step,
                        "activation_rms": float(state.float().square().mean().sqrt().item()),
                        "gradient_rms": 0.0,
                        "gradient_activation_rms": 0.0,
                    }
                )
                continue
            rows.append(
                {
                    "step": step,
                    "activation_rms": float(state.float().square().mean().sqrt().item()),
                    "gradient_rms": float(
                        gradient.float().square().mean().sqrt().item()
                    ),
                    "gradient_activation_rms": float(
                        (gradient.float() * state.float()).square().mean().sqrt().item()
                    ),
                }
            )
        return rows

    residue_count = len(trace.residue_states)
    return {
        "loss": float(loss.item()),
        "residue": summarize(
            trace.residue_states,
            gradients[:residue_count],
        ),
        "scratch": summarize(
            scratch_sources,
            gradients[residue_count:],
        ),
    }


def _prompt_number(input_ids: Tensor, marker: int) -> Tensor:
    values = torch.zeros(
        input_ids.shape[0],
        dtype=torch.long,
        device=input_ids.device,
    )
    reading = torch.zeros_like(values, dtype=torch.bool)
    for position in range(input_ids.shape[1]):
        token = input_ids[:, position]
        reading = torch.where(
            token == marker,
            torch.ones_like(reading),
            reading,
        )
        digit = token >= DIGIT_OFFSET
        values = torch.where(
            reading & digit,
            values * 10 + token - DIGIT_OFFSET,
            values,
        )
        reading = reading & ((token == marker) | digit)
    return values


def _iteration_readouts(
    model,
    states: tuple[Tensor, ...],
    input_ids: Tensor,
    attention_mask: Tensor,
) -> list[dict[str, float | int]]:
    moduli = _prompt_number(input_ids, TOKEN_IDS["N"])
    expected_values = _prompt_number(input_ids, TOKEN_IDS["X"])
    requested_steps = _prompt_number(input_ids, TOKEN_IDS["T"])
    valid_lengths = attention_mask.long().sum(dim=-1)
    readouts = []

    for step, state in enumerate(states):
        if step:
            expected_values = expected_values.square().remainder(moduli)
        eligible = requested_steps >= step
        logits = model.decode_residue(
            state,
            attention_mask,
            input_ids.shape[1],
        ).float()
        losses = []
        exact = []
        token_count = 0
        for row in eligible.nonzero(as_tuple=False).flatten():
            expected_tokens = torch.tensor(
                number_tokens(int(expected_values[row].item())),
                dtype=torch.long,
                device=logits.device,
            )
            end = int(valid_lengths[row].item())
            start = end - expected_tokens.numel()
            token_logits = logits[row, start:end]
            losses.append(
                F.cross_entropy(
                    token_logits,
                    expected_tokens,
                    reduction="sum",
                )
            )
            token_count += expected_tokens.numel()
            exact.append(
                bool(
                    torch.equal(
                        token_logits.argmax(dim=-1),
                        expected_tokens,
                    )
                )
            )
        if not exact:
            continue
        readouts.append(
            {
                "step": step,
                "exact_accuracy": sum(exact) / len(exact),
                "correct_examples": sum(exact),
                "example_count": len(exact),
                "token_cross_entropy": float(
                    (torch.stack(losses).sum() / token_count).item()
                ),
            }
        )
    return readouts


def _load_trained_model(run_dir: Path, device: torch.device):
    manifest_path = run_dir / "manifest.json"
    submission_path = run_dir / "submission.py"
    manifest = load_manifest(manifest_path)
    checkpoint_paths = sorted(run_dir.glob("checkpoint-seed-*.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError("analysis requires exactly one checkpoint in the run directory")
    checkpoint = torch.load(checkpoint_paths[0], map_location=device, weights_only=True)
    submission = _load_submission_file(submission_path)
    model = submission.build_model(_make_model_spec(manifest))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    if not hasattr(model, "forward_with_trace"):
        raise TypeError("trained model does not expose forward_with_trace")
    return manifest, checkpoint, model


def _trace_model(model, batch, device: torch.device):
    input_ids, targets, attention_mask, target_positions = prepare_batch(batch, device)
    if target_positions is None:
        raise ValueError("geometry analysis currently requires separate output positions")
    with torch.no_grad():
        trace = model.forward_with_trace(input_ids, attention_mask=attention_mask)
    return trace, targets, attention_mask, target_positions


def analyze_checkpoint(
    *,
    run_dir: Path,
    split: str,
    output_path: Path,
    max_examples: int,
    reference_run_dir: Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest, checkpoint, model = _load_trained_model(run_dir, device)
    seed = int(checkpoint["seed"])
    data_config = replace(
        manifest.data,
        seed=seed,
        batch_size=max_examples,
        eval_batch_size=max_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dataloaders = make_dataloaders(data_config, device=device)
    if split not in dataloaders:
        raise ValueError(f"unknown split {split!r}; available={sorted(dataloaders)}")
    batch = next(iter(dataloaders[split]))
    trace, targets, attention_mask, target_positions = _trace_model(model, batch, device)
    uses_residue_state = bool(trace.residue_states)
    if uses_residue_state:
        states = trace.residue_states
        state_mask = torch.ones(
            states[0].shape[:2],
            dtype=torch.bool,
            device=states[0].device,
        )
        state_channel = "residue"
    else:
        states = trace.prompt_states
        state_mask = attention_mask
        state_channel = "prompt"
    if not states:
        raise ValueError("trained model trace contains no recurrent states")
    flattened = [_flatten_valid(state, state_mask) for state in states]

    readouts = []
    for index, hidden in enumerate(states):
        metrics = (
            _residue_readout_metrics(
                model,
                hidden,
                targets,
                target_positions,
                attention_mask,
            )
            if uses_residue_state
            else _readout_metrics(model, hidden, targets, target_positions)
        )
        readouts.append(
            {
                "step": index,
                **metrics,
            }
        )

    transitions = []
    for index in range(1, len(flattened)):
        previous = flattened[index - 1]
        current = flattened[index]
        transitions.append(
            {
                "from_step": index - 1,
                "to_step": index,
                **_transition_metrics(previous, current),
            }
        )

    lag_two = []
    for index in range(2, len(flattened)):
        previous = flattened[index - 2]
        current = flattened[index]
        lag_two.append(
            {
                "from_step": index - 2,
                "to_step": index,
                **_transition_metrics(previous, current),
            }
        )

    register_transitions = []
    if not uses_residue_state:
        for index in range(1, len(trace.register_states)):
            previous = trace.register_states[index - 1].argmax(dim=-1)
            current = trace.register_states[index].argmax(dim=-1)
            register_transitions.append(
                {
                    "from_step": index - 1,
                    "to_step": index,
                    "changed_fraction": float(
                        (previous != current).float().mean().item()
                    ),
                }
            )

    register_lag_two = []
    if not uses_residue_state:
        for index in range(2, len(trace.register_states)):
            previous = trace.register_states[index - 2].argmax(dim=-1)
            current = trace.register_states[index].argmax(dim=-1)
            register_lag_two.append(
                {
                    "from_step": index - 2,
                    "to_step": index,
                    "returned_fraction": float(
                        (previous == current).float().mean().item()
                    ),
                }
            )

    uses_scratch_state = bool(trace.scratch_states) and bool(
        trace.scratch_states[0].numel()
    )
    scratch_transitions = []
    if uses_residue_state and uses_scratch_state:
        flattened_scratch = [
            state.reshape(-1, state.shape[-1]) for state in trace.scratch_states
        ]
        for index in range(1, len(flattened_scratch)):
            scratch_transitions.append(
                {
                    "from_step": index - 1,
                    "to_step": index,
                    **_transition_metrics(
                        flattened_scratch[index - 1],
                        flattened_scratch[index],
                    ),
                }
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "manifest": manifest.name,
        "seed": seed,
        "split": split,
        "state_channel": state_channel,
        "sample_examples": readouts[0]["example_count"],
        "recurrent_steps": len(states) - 1,
        "readouts": readouts,
        "iteration_readouts": (
            _iteration_readouts(
                model,
                states,
                batch["input_ids"].to(device),
                batch.get("attention_mask", batch["input_ids"] != 0).to(device),
            )
            if uses_residue_state
            else []
        ),
        "transitions": transitions,
        "lag_two": lag_two,
        "register_transitions": register_transitions,
        "register_lag_two": register_lag_two,
        "scratch_transitions": scratch_transitions,
        "scratch_statistics": (
            _state_statistics(trace.scratch_states)
            if uses_residue_state and uses_scratch_state
            else []
        ),
        "gradient_sensitivity": (
            _gradient_sensitivity(model, batch, device)
            if uses_residue_state
            else None
        ),
        "pair_routing": _pair_routing_statistics(model),
    }

    if reference_run_dir is not None:
        reference_manifest, reference_checkpoint, reference_model = _load_trained_model(
            reference_run_dir,
            device,
        )
        if reference_manifest.name != manifest.name:
            raise ValueError("cross-model geometry requires matching manifests")
        reference_trace, _, _, _ = _trace_model(reference_model, batch, device)
        reference_states = (
            reference_trace.residue_states
            if uses_residue_state
            else reference_trace.prompt_states
        )
        reference_mask = (
            torch.ones(
                reference_states[0].shape[:2],
                dtype=torch.bool,
                device=reference_states[0].device,
            )
            if uses_residue_state
            else attention_mask
        )
        reference_flattened = [
            _flatten_valid(state, reference_mask) for state in reference_states
        ]
        if len(reference_flattened) != len(flattened):
            raise ValueError("cross-model geometry requires matching recurrent steps")
        result["reference"] = {
            "run_id": reference_run_dir.name,
            "seed": int(reference_checkpoint["seed"]),
            "step_alignment": [
                {
                    "step": index,
                    "linear_cka": linear_cka(current, reference),
                    "procrustes_similarity": orthogonal_procrustes_similarity(
                        current,
                        reference,
                    ),
                }
                for index, (current, reference) in enumerate(
                    zip(flattened, reference_flattened, strict=True)
                )
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"GEOMETRY_RESULT={output_path}", flush=True)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--reference-run-dir", type=Path)
    args = parser.parse_args()
    analyze_checkpoint(
        run_dir=args.run_dir,
        split=args.split,
        output_path=args.output,
        max_examples=args.max_examples,
        reference_run_dir=args.reference_run_dir,
    )


if __name__ == "__main__":
    cli()
