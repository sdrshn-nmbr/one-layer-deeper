"""Fit a recurrent Jacobian lens and inspect local transition spectra."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from benchmark.batches import prepare_batch
from data import make_dataloaders
from research.geometry import (
    _load_trained_model,
    _readout_metrics,
    _residue_readout_metrics,
    linear_cka,
    orthogonal_procrustes_similarity,
)


def average_jacobians(
    *,
    target: Tensor,
    sources: list[Tensor],
    attention_mask: Tensor,
    original_batch_size: int,
    dim_batch: int,
) -> list[Tensor]:
    """Estimate average output-to-input Jacobians with batched cotangents."""

    width = target.shape[-1]
    length = target.shape[1]
    if target.shape[0] != original_batch_size * dim_batch:
        raise ValueError("target batch does not match replicated cotangent batch")
    if attention_mask.shape != (original_batch_size, length):
        raise ValueError("attention mask does not match the unreplicated batch")
    valid = attention_mask.to(device=target.device, dtype=torch.bool)
    jacobians = [torch.zeros(width, width, dtype=torch.float32) for _ in sources]
    passes = math.ceil(width / dim_batch)
    for pass_index, start in enumerate(range(0, width, dim_batch)):
        count = min(dim_batch, width - start)
        cotangent = torch.zeros_like(target)
        cotangent = cotangent.view(
            dim_batch,
            original_batch_size,
            length,
            width,
        )
        for replica in range(count):
            cotangent[replica, :, :, start + replica] = valid
        gradients = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=cotangent.view_as(target),
            retain_graph=True,
        )
        for index, gradient in enumerate(gradients):
            gradient = gradient.view(
                dim_batch,
                original_batch_size,
                length,
                width,
            )
            rows = gradient[:count][:, valid, :].float().mean(dim=1)
            jacobians[index][start : start + count] = rows.detach().cpu()
        if pass_index == passes - 1:
            del gradients
    return jacobians


def _spectrum(matrix: Tensor) -> dict[str, float | int]:
    singular_values = torch.linalg.svdvals(matrix)
    eigenvalues = torch.linalg.eigvals(matrix)
    distances = (eigenvalues + 1).abs()
    return {
        "frobenius_over_sqrt_width": float(
            (torch.linalg.vector_norm(matrix) / math.sqrt(matrix.shape[0])).item()
        ),
        "largest_singular_value": float(singular_values.max().item()),
        "median_singular_value": float(singular_values.median().item()),
        "spectral_radius": float(eigenvalues.abs().max().item()),
        "most_negative_real_eigenvalue": float(eigenvalues.real.min().item()),
        "nearest_minus_one_distance": float(distances.min().item()),
        "near_minus_one_count": int((distances < 0.1).sum().item()),
    }


def _replicated_trace(model, batch, device: torch.device, dim_batch: int):
    input_ids, targets, attention_mask, target_positions = prepare_batch(batch, device)
    if target_positions is None:
        raise ValueError("Jacobian analysis requires separate output positions")
    original_batch_size = input_ids.shape[0]
    replicated_ids = input_ids.repeat(dim_batch, 1)
    replicated_mask = attention_mask.repeat(dim_batch, 1)
    with torch.enable_grad():
        trace = model.forward_with_trace(
            replicated_ids,
            attention_mask=replicated_mask,
        )
    state_mask = (
        torch.ones(
            original_batch_size,
            trace.residue_states[0].shape[1],
            dtype=torch.bool,
            device=device,
        )
        if trace.residue_states
        else attention_mask
    )
    return (
        trace,
        targets,
        state_mask,
        target_positions,
        original_batch_size,
    )


def _ordinary_trace(model, batch, device: torch.device):
    input_ids, targets, attention_mask, target_positions = prepare_batch(batch, device)
    if target_positions is None:
        raise ValueError("Jacobian analysis requires separate output positions")
    with torch.no_grad():
        trace = model.forward_with_trace(input_ids, attention_mask=attention_mask)
    return trace, targets, target_positions, attention_mask


def _trace_states(trace) -> tuple[tuple[Tensor, ...], str]:
    if trace.residue_states:
        return trace.residue_states, "residue"
    if trace.prompt_states:
        return trace.prompt_states, "prompt"
    raise ValueError("Jacobian analysis requires recurrent trace states")


def _direct_readout(
    model,
    state: Tensor,
    targets: Tensor,
    target_positions: Tensor,
    attention_mask: Tensor,
    state_channel: str,
) -> dict[str, float | int]:
    if state_channel == "residue":
        return _residue_readout_metrics(
            model,
            state,
            targets,
            target_positions,
            attention_mask,
        )
    return _readout_metrics(model, state, targets, target_positions)


def analyze_jacobian_lens(
    *,
    run_dir: Path,
    split: str,
    output_path: Path,
    fit_examples: int,
    dim_batch: int,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest, checkpoint, model = _load_trained_model(run_dir, device)
    seed = int(checkpoint["seed"])
    data_config = replace(
        manifest.data,
        seed=seed,
        batch_size=fit_examples,
        eval_batch_size=fit_examples,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dataloaders = make_dataloaders(data_config, device=device)
    if split not in dataloaders:
        raise ValueError(f"unknown split {split!r}; available={sorted(dataloaders)}")
    iterator = iter(dataloaders[split])
    fit_batch = next(iterator)
    try:
        holdout_batch = next(iterator)
    except StopIteration as error:
        raise ValueError("split needs at least two batches for lens fit and holdout") from error

    (
        fit_trace,
        _,
        fit_mask,
        _,
        original_batch_size,
    ) = _replicated_trace(model, fit_batch, device, dim_batch)
    fit_states, state_channel = _trace_states(fit_trace)
    final_step = len(fit_states) - 1
    source_steps = sorted(
        {
            0,
            min(1, final_step),
            min(2, final_step),
            max(0, final_step - 1),
        }
    )
    source_states = [fit_states[index] for index in source_steps]
    final_jacobians = average_jacobians(
        target=fit_states[-1],
        sources=source_states,
        attention_mask=fit_mask,
        original_batch_size=original_batch_size,
        dim_batch=dim_batch,
    )

    local_pairs = [(index, index + 1) for index in range(final_step)]
    local_jacobians = []
    for source_step, target_step in local_pairs:
        local_jacobians.append(
            average_jacobians(
                target=fit_states[target_step],
                sources=[fit_states[source_step]],
                attention_mask=fit_mask,
                original_batch_size=original_batch_size,
                dim_batch=dim_batch,
            )[0]
        )

    (
        holdout_trace,
        holdout_targets,
        holdout_positions,
        holdout_attention_mask,
    ) = _ordinary_trace(
        model,
        holdout_batch,
        device,
    )
    holdout_states, holdout_channel = _trace_states(holdout_trace)
    if holdout_channel != state_channel:
        raise ValueError("fit and holdout traces use different state channels")
    lens_readouts = []
    for source_step, jacobian in zip(source_steps, final_jacobians, strict=True):
        source = holdout_states[source_step]
        transported = torch.einsum(
            "...d,od->...o",
            source.float(),
            jacobian.to(device),
        )
        lens_readouts.append(
            {
                "source_step": source_step,
                "direct": _direct_readout(
                    model,
                    source,
                    holdout_targets,
                    holdout_positions,
                    holdout_attention_mask,
                    state_channel,
                ),
                "jacobian_lens": _direct_readout(
                    model,
                    transported,
                    holdout_targets,
                    holdout_positions,
                    holdout_attention_mask,
                    state_channel,
                ),
                "spectrum": _spectrum(jacobian),
            }
        )

    local = [
        {
            "source_step": source_step,
            "target_step": target_step,
            "spectrum": _spectrum(jacobian),
        }
        for (source_step, target_step), jacobian in zip(
            local_pairs,
            local_jacobians,
            strict=True,
        )
    ]
    local_alignment = [
        {
            "left_transition": list(local_pairs[index - 1]),
            "right_transition": list(local_pairs[index]),
            "linear_cka": linear_cka(
                local_jacobians[index - 1],
                local_jacobians[index],
            ),
            "procrustes_similarity": orthogonal_procrustes_similarity(
                local_jacobians[index - 1],
                local_jacobians[index],
            ),
        }
        for index in range(1, len(local_jacobians))
    ]

    result = {
        "schema_version": 1,
        "method": "average_jacobian_lens_recurrent_state_mean",
        "run_id": run_dir.name,
        "manifest": manifest.name,
        "seed": seed,
        "split": split,
        "fit_examples": original_batch_size,
        "holdout_examples": int(holdout_targets.shape[0]),
        "dim_batch": dim_batch,
        "final_step": final_step,
        "state_channel": state_channel,
        "final_readout": _direct_readout(
            model,
            holdout_states[-1],
            holdout_targets,
            holdout_positions,
            holdout_attention_mask,
            state_channel,
        ),
        "lens_readouts": lens_readouts,
        "local_transition_jacobians": local,
        "local_transition_alignment": local_alignment,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"JACOBIAN_RESULT={output_path}", flush=True)
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-examples", type=int, default=32)
    parser.add_argument("--dim-batch", type=int, default=16)
    args = parser.parse_args()
    analyze_jacobian_lens(
        run_dir=args.run_dir,
        split=args.split,
        output_path=args.output,
        fit_examples=args.fit_examples,
        dim_batch=args.dim_batch,
    )


if __name__ == "__main__":
    cli()
