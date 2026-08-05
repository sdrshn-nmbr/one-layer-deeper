"""Synchronously benchmark exact evaluator training steps on one CUDA device."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import traceback
from typing import Any, Callable
import uuid

import torch

from benchmark import OptimizerSpec
from benchmark.manifest import load_manifest
from benchmark.runner import (
    _configure_seed,
    _load_submission_file,
    _loss_and_accuracy,
    _make_model_spec,
    _next_batch,
    _validate_model_interface,
)
from benchmark.validation import validate_model_state, validate_optimizer
from data import make_dataloaders
from research.runner import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_SUBMISSION,
    _atomic_json,
    _environment_receipt,
    _file_record,
    _utc_now,
    materialize_submission,
)


def summarize_samples(samples: list[float]) -> dict[str, float | int]:
    if len(samples) < 3:
        raise ValueError("timing evidence requires at least three samples")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in samples
    ):
        raise ValueError("timing samples must be positive finite numbers")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    return {
        "count": len(samples),
        "minimum_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "maximum_ms": max(samples),
        "standard_deviation_ms": deviation,
        "coefficient_of_variation": deviation / mean,
    }


def compile_break_even(
    *,
    budget_seconds: float,
    eager_step_ms: float,
    compiled_step_ms: float,
    compile_overhead_ms: float,
) -> dict[str, float | bool]:
    values = (
        budget_seconds,
        eager_step_ms,
        compiled_step_ms,
        compile_overhead_ms,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("break-even inputs must be positive finite numbers")
    budget_ms = budget_seconds * 1000
    eager_updates = budget_ms / eager_step_ms
    candidate_updates = max(0.0, budget_ms - compile_overhead_ms) / compiled_step_ms
    maximum_overhead_ms = max(
        0.0,
        budget_ms * (1 - compiled_step_ms / eager_step_ms),
    )
    return {
        "eager_updates": eager_updates,
        "candidate_updates": candidate_updates,
        "maximum_compile_overhead_ms": maximum_overhead_ms,
        "promoted": candidate_updates > eager_updates,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_dtype(manifest) -> torch.dtype:
    if manifest.runtime.amp:
        return torch.float32
    return getattr(torch, manifest.runtime.dtype)


def _run_step(
    *,
    model,
    bundle,
    batch,
    manifest,
    device: torch.device,
    training_loss,
    token_training_loss,
) -> tuple[float, float]:
    optimizer = bundle.optimizer
    optimizer.zero_grad(set_to_none=True)
    loss, accuracy, _, _ = _loss_and_accuracy(
        model,
        batch,
        manifest,
        device,
        training_loss=training_loss,
        token_training_loss=token_training_loss,
    )
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("benchmark step produced non-finite loss")
    loss.backward()
    if manifest.runtime.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            manifest.runtime.grad_clip,
        )
    optimizer.step()
    if bundle.scheduler is not None:
        bundle.scheduler.step()
    return float(loss.item()), accuracy


def _time_complete_step(
    *,
    model,
    bundle,
    iterator,
    dataloader,
    manifest,
    device: torch.device,
    training_loss,
    token_training_loss,
) -> tuple[dict[str, float], Any, float, float]:
    _synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    validate_model_state(model, manifest.model_state, device)
    batch, iterator = _next_batch(iterator, dataloader)
    loss, accuracy = _run_step(
        model=model,
        bundle=bundle,
        batch=batch,
        manifest=manifest,
        device=device,
        training_loss=training_loss,
        token_training_loss=token_training_loss,
    )
    end_event.record()
    _synchronize(device)
    host_ms = (time.perf_counter() - started) * 1000
    cuda_ms = start_event.elapsed_time(end_event)
    return {"host_ms": host_ms, "cuda_ms": cuda_ms}, iterator, loss, accuracy


def _time_stage(
    device: torch.device,
    operation: Callable[[], Any],
) -> tuple[float, Any]:
    _synchronize(device)
    started = time.perf_counter()
    result = operation()
    _synchronize(device)
    return (time.perf_counter() - started) * 1000, result


def _stage_sample(
    *,
    model,
    bundle,
    iterator,
    dataloader,
    manifest,
    device: torch.device,
    training_loss,
    token_training_loss,
) -> tuple[dict[str, float], Any]:
    optimizer = bundle.optimizer
    stages: dict[str, float] = {}
    stages["validate_model_ms"], _ = _time_stage(
        device,
        lambda: validate_model_state(model, manifest.model_state, device),
    )
    stages["batch_fetch_ms"], fetched = _time_stage(
        device,
        lambda: _next_batch(iterator, dataloader),
    )
    batch, iterator = fetched
    stages["zero_grad_ms"], _ = _time_stage(
        device,
        lambda: optimizer.zero_grad(set_to_none=True),
    )
    stages["forward_loss_ms"], forward = _time_stage(
        device,
        lambda: _loss_and_accuracy(
            model,
            batch,
            manifest,
            device,
            training_loss=training_loss,
            token_training_loss=token_training_loss,
        ),
    )
    loss, _, _, _ = forward
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError("stage benchmark produced non-finite loss")
    stages["backward_ms"], _ = _time_stage(device, loss.backward)
    if manifest.runtime.grad_clip is not None:
        stages["grad_clip_ms"], _ = _time_stage(
            device,
            lambda: torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                manifest.runtime.grad_clip,
            ),
        )
    stages["optimizer_ms"], _ = _time_stage(device, optimizer.step)
    if bundle.scheduler is not None:
        stages["scheduler_ms"], _ = _time_stage(
            device,
            bundle.scheduler.step,
        )
    return stages, iterator


def benchmark_experiment(
    *,
    submission_path: Path,
    experiment: str,
    manifest_path: Path,
    artifact_root: Path,
    warmup_steps: int,
    sample_steps: int,
    stage_samples: int,
) -> Path:
    if warmup_steps < 0 or sample_steps < 3 or stage_samples < 3:
        raise ValueError("warmup must be non-negative; sample counts must be at least three")
    manifest = load_manifest(manifest_path)
    device = torch.device(manifest.runtime.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("step benchmark requires one available CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("step benchmark requires exactly one visible CUDA device")
    torch.cuda.set_device(device)
    seed = manifest.runtime.seeds[0]
    _configure_seed(seed, device)

    root = artifact_root.resolve() / "performance"
    root.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-"
        f"{experiment}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "synchronized_training_step_benchmark",
        "run_id": run_id,
        "experiment": experiment,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "environment": _environment_receipt(),
        "artifacts": {},
    }
    _atomic_json(run_dir / "receipt.json", receipt)

    try:
        exact_submission = materialize_submission(
            submission_path,
            experiment,
            run_dir / "submission.py",
        )
        shutil.copyfile(manifest_path, run_dir / "manifest.json")
        dataloaders = make_dataloaders(
            replace(manifest.data, seed=seed),
            device=device,
        )
        dataloader = dataloaders["train"]
        iterator = iter(dataloader)
        model_spec = _make_model_spec(manifest)

        _synchronize(device)
        setup_started = time.perf_counter()
        submission = _load_submission_file(exact_submission)
        model = submission.build_model(model_spec)
        model = model.to(device=device, dtype=_model_dtype(manifest))
        _validate_model_interface(model, model_spec)
        validate_model_state(model, manifest.model_state, device)
        bundle = submission.build_optimizer(
            model,
            OptimizerSpec(
                training_time_seconds=manifest.runtime.total_training_time_seconds,
                device_type=device.type,
            ),
        )
        validate_optimizer(bundle, model, device)
        _synchronize(device)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        model.train()

        cold, iterator, last_loss, last_accuracy = _time_complete_step(
            model=model,
            bundle=bundle,
            iterator=iterator,
            dataloader=dataloader,
            manifest=manifest,
            device=device,
            training_loss=submission.training_loss,
            token_training_loss=submission.token_training_loss,
        )
        warmup: list[dict[str, float]] = []
        for _ in range(warmup_steps):
            sample, iterator, last_loss, last_accuracy = _time_complete_step(
                model=model,
                bundle=bundle,
                iterator=iterator,
                dataloader=dataloader,
                manifest=manifest,
                device=device,
                training_loss=submission.training_loss,
                token_training_loss=submission.token_training_loss,
            )
            warmup.append(sample)

        clean: list[dict[str, float]] = []
        for _ in range(sample_steps):
            sample, iterator, last_loss, last_accuracy = _time_complete_step(
                model=model,
                bundle=bundle,
                iterator=iterator,
                dataloader=dataloader,
                manifest=manifest,
                device=device,
                training_loss=submission.training_loss,
                token_training_loss=submission.token_training_loss,
            )
            clean.append(sample)

        stages: list[dict[str, float]] = []
        for _ in range(stage_samples):
            sample, iterator = _stage_sample(
                model=model,
                bundle=bundle,
                iterator=iterator,
                dataloader=dataloader,
                manifest=manifest,
                device=device,
                training_loss=submission.training_loss,
                token_training_loss=submission.token_training_loss,
            )
            stages.append(sample)

        stage_names = sorted(stages[0])
        result = {
            "schema_version": 1,
            "experiment": experiment,
            "manifest": manifest.name,
            "seed": seed,
            "setup_in_budget_ms": setup_ms,
            "cold_step": cold,
            "warmup_steps": warmup,
            "clean_steps": clean,
            "clean_summary": {
                "host": summarize_samples([sample["host_ms"] for sample in clean]),
                "cuda": summarize_samples([sample["cuda_ms"] for sample in clean]),
            },
            "stage_samples": stages,
            "stage_summary": {
                name: summarize_samples([sample[name] for sample in stages])
                for name in stage_names
            },
            "final_loss": last_loss,
            "final_accuracy": last_accuracy,
            "model_state_elements": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
        _atomic_json(run_dir / "result.json", result)
        artifact_names = ("manifest.json", "result.json", "submission.py")
        receipt.update(
            {
                "status": "succeeded",
                "finished_at": _utc_now(),
                "artifacts": {
                    name: _file_record(run_dir / name) for name in artifact_names
                },
            }
        )
        _atomic_json(run_dir / "receipt.json", receipt)
    except Exception as error:
        receipt.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        _atomic_json(run_dir / "receipt.json", receipt)
        raise

    print("PERF_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    print(f"PERF_ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--stage-samples", type=int, default=5)
    args = parser.parse_args()
    benchmark_experiment(
        submission_path=args.submission,
        experiment=args.experiment,
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        warmup_steps=args.warmup_steps,
        sample_steps=args.sample_steps,
        stage_samples=args.stage_samples,
    )


if __name__ == "__main__":
    cli()
