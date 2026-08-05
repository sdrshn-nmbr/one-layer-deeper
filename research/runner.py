"""Run a materialized model variant through the official benchmark evaluator."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from threading import Event, Thread
import traceback
from typing import Any, TextIO
from unittest.mock import patch
import uuid

import torch

from benchmark.runner import _make_model_spec, _load_submission_file, run_submission_file
from benchmark.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "submissions" / "recurrent" / "submission.py"
DEFAULT_MANIFEST = ROOT / "benchmark" / "manifests" / "smoke_cpu.json"
DEFAULT_ARTIFACT_ROOT = ROOT / ".artifacts"
EXPERIMENT_PATTERN = re.compile(
    r'^(SELECTED_EXPERIMENT\s*=\s*)["\'][^"\']+["\']\s*$',
    re.MULTILINE,
)


class Tee:
    """Copy evaluator stdout to the console and the run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class GpuSampler:
    """Sample device-level GPU counters without changing the evaluator loop."""

    FIELDS = (
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.sm",
        "temperature.gpu",
    )

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool,
        interval_seconds: float = 0.25,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.samples = 0
        self.status = "disabled"
        with path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(("sampled_at_utc", *self.FIELDS))

    def _sample(self) -> list[str]:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--query-gpu={','.join(self.FIELDS)}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        rows = list(csv.reader(completed.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(self.FIELDS):
            raise ValueError("nvidia-smi returned an unexpected row shape")
        return [value.strip() for value in rows[0]]

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                while not self.stop_event.is_set():
                    writer.writerow((_utc_now(), *self._sample()))
                    file.flush()
                    self.samples += 1
                    self.stop_event.wait(self.interval_seconds)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.status = "failed"
            print(f"[gpu-sampler] failed: {type(error).__name__}: {error}", file=sys.stderr)

    def __enter__(self) -> GpuSampler:
        if not self.enabled:
            print("[gpu-sampler] disabled by configuration", flush=True)
            return self
        if not torch.cuda.is_available() or shutil.which("nvidia-smi") is None:
            print("[gpu-sampler] disabled: CUDA or nvidia-smi unavailable", flush=True)
            return self
        self.status = "running"
        self.thread = Thread(target=self._run, name="gpu-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.status == "running":
            self.status = "succeeded"
        print(
            f"[gpu-sampler] status={self.status} samples={self.samples}",
            flush=True,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _environment_receipt() -> dict[str, Any]:
    device: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        device.update(
            {
                "cuda_device_name": properties.name,
                "cuda_total_memory_bytes": properties.total_memory,
                "cuda_capability": [properties.major, properties.minor],
            }
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "git": {
            "revision": _git_value("rev-parse", "HEAD"),
            "status_porcelain": _git_value("status", "--porcelain"),
        },
    }


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"research_materialized_{path.stat().st_mtime_ns}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import materialized submission at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_submission(
    source: str | Path,
    experiment: str,
    destination: str | Path,
) -> Path:
    """Write the exact standalone source used for one named experiment."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    text = source_path.read_text()
    materialized, replacements = EXPERIMENT_PATTERN.subn(
        rf'\1"{experiment}"', text
    )
    if replacements != 1:
        raise ValueError(
            "submission source must contain exactly one SELECTED_EXPERIMENT assignment"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(materialized)
    return destination_path


def _calculated_workload(
    submission_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Record a transparent FLOP and state-size estimate, not measured profiling."""

    manifest = load_manifest(manifest_path)
    model_spec = _make_model_spec(manifest)
    module = _load_module(submission_path)
    experiment = module.EXPERIMENTS[module.SELECTED_EXPERIMENT]
    config = experiment.model
    submission = _load_submission_file(submission_path)
    batch_size = submission.batch_size or manifest.data.batch_size
    prompt_tokens = model_spec.max_seq_len
    total_tokens = prompt_tokens + config.state_tokens
    loops = config.num_loops
    training_loops = config.training_loop_cap or loops
    layers_per_loop = config.step_layers
    block_applications = training_loops * layers_per_loop
    d_model = config.d_model
    ratio = config.mlp_ratio
    vocabulary = model_spec.vocab_size

    output_head = 2 * batch_size * prompt_tokens * d_model * vocabulary
    if config.architecture == "causal_state":
        digit_slots = config.digit_slots
        scratch_tokens = config.scratch_tokens
        scratch_projection = (
            4 * batch_size * d_model**2 if scratch_tokens else 0
        )
        scratch_gru = 12 * batch_size * scratch_tokens * d_model**2
        residue_projection = 4 * batch_size * digit_slots * d_model**2
        residue_gru = 12 * batch_size * digit_slots * d_model**2
        recurrent_step = (
            scratch_projection
            + scratch_gru
            + residue_projection
            + residue_gru
        )
        attention = 0
        forward_flops = training_loops * recurrent_step + output_head
    else:
        projections_and_mlp = (
            (8 + 4 * ratio) * batch_size * total_tokens * d_model**2
        )
        attention = 4 * batch_size * total_tokens**2 * d_model
        forward_flops = (
            block_applications * (projections_and_mlp + attention) + output_head
        )

    model = submission.build_model(model_spec)
    model_elements = sum(parameter.numel() for parameter in model.parameters())
    return {
        "kind": "calculated_upper_level_estimate",
        "execution": {
            "optimizer_implementation": experiment.optimizer.implementation,
            "max_batch_uses": experiment.optimizer.max_batch_uses,
            "compile_model": experiment.compile_model,
            "wall_clock_schedule": experiment.optimizer.wall_clock_schedule,
        },
        "inputs": {
            "architecture": config.architecture,
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "state_tokens": config.state_tokens,
            "scratch_tokens": config.scratch_tokens,
            "digit_slots": config.digit_slots,
            "total_tokens": total_tokens,
            "d_model": d_model,
            "ffn_multiplier": ratio,
            "recurrent_loops": loops,
            "training_recurrent_loops": training_loops,
            "layers_per_loop": layers_per_loop,
            "state_feedback": config.state_feedback,
            "final_hidden_mode": config.final_hidden_mode,
            "structured_features": config.structured_features,
            "training_loop_cap": config.training_loop_cap,
            "vocab_size": vocabulary,
        },
        "flops": {
            "forward": forward_flops,
            "forward_and_backward_approx": 3 * forward_flops,
            "attention_forward": block_applications * attention,
            "attention_fraction_of_forward": (
                block_applications * attention / forward_flops
            ),
        },
        "state": {
            "model_elements": model_elements,
            "model_bytes_fp32": 4 * model_elements,
            "adamw_model_gradient_optimizer_bytes_fp32_approx": 16 * model_elements,
        },
        "assumptions": [
            "one multiply-add counts as two FLOPs",
            "backward is approximated as twice forward",
            "embedding lookup, normalization, activation, and optimizer FLOPs are omitted",
            "the estimate uses the configured training loop cap; actual prompt T may be smaller",
            "the causal-state GRU estimate omits elementwise gates and reductions",
            "this file is calculated; measured device profiling is separate evidence",
        ],
    }


def _artifact_inventory(run_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "gpu_samples.csv",
        "manifest.json",
        "result.json",
        "stdout.log",
        "submission.py",
        "workload.json",
    ]
    names.extend(
        path.name for path in sorted(run_dir.glob("checkpoint-seed-*.pt"))
    )
    return {name: _file_record(run_dir / name) for name in names}


def run_submission_file_with_checkpoints(
    submission_path: Path,
    manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Run the official evaluator and save its final trained model afterward."""

    submission = _load_submission_file(submission_path)
    trained_models: list[torch.nn.Module] = []
    original_build_model = submission.build_model

    def build_and_capture(spec):
        model = original_build_model(spec)
        trained_models.append(model)
        return model

    instrumented = replace(submission, build_model=build_and_capture)
    with patch(
        "benchmark.runner._load_submission_file",
        return_value=instrumented,
    ):
        result = run_submission_file(
            submission_path,
            manifest_path,
            include_structured_metrics=True,
        )

    seeds = result.get("seeds", [])
    if len(trained_models) != len(seeds):
        raise RuntimeError(
            "checkpoint capture model count does not match evaluator seed count"
        )
    for model, seed_result in zip(trained_models, seeds, strict=True):
        seed = int(seed_result["seed"])
        state_dict = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
        torch.save(
            {
                "schema_version": 1,
                "seed": seed,
                "submission_sha256": _sha256(submission_path),
                "manifest_sha256": _sha256(manifest_path),
                "score": result["score"],
                "state_dict": state_dict,
            },
            run_dir / f"checkpoint-seed-{seed}.pt",
        )
    return result


def verify_run_artifacts(run_dir: str | Path) -> bool:
    """Return true only for a successful receipt whose artifacts still match."""

    directory = Path(run_dir).resolve()
    try:
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt.get("status") != "succeeded":
            return False
        artifacts = receipt["artifacts"]
        for name, expected in artifacts.items():
            path = directory / name
            if not path.is_file() or _file_record(path) != expected:
                return False
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def run_experiment(
    *,
    submission_path: str | Path,
    experiment: str,
    manifest_path: str | Path,
    artifact_root: str | Path,
    sample_gpu: bool = False,
    save_checkpoint: bool = False,
) -> Path:
    """Materialize, evaluate, and seal one experiment run."""

    source = Path(submission_path).resolve()
    manifest = Path(manifest_path).resolve()
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{experiment}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir()
    receipt: dict[str, Any] = {
        "schema_version": 1,
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
            source,
            experiment,
            run_dir / "submission.py",
        )
        shutil.copyfile(manifest, run_dir / "manifest.json")
        _atomic_json(
            run_dir / "workload.json",
            _calculated_workload(exact_submission, manifest),
        )
        with GpuSampler(
            run_dir / "gpu_samples.csv",
            enabled=sample_gpu,
        ) as gpu_sampler:
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(Tee(sys.stdout, log)):
                    if save_checkpoint:
                        result = run_submission_file_with_checkpoints(
                            exact_submission,
                            manifest,
                            run_dir,
                        )
                    else:
                        result = run_submission_file(
                            exact_submission,
                            manifest,
                            include_structured_metrics=True,
                        )
        _atomic_json(run_dir / "result.json", result)
        receipt.update(
            {
                "status": "succeeded",
                "finished_at": _utc_now(),
                "gpu_sampling": {
                    "status": gpu_sampler.status,
                    "interval_seconds": gpu_sampler.interval_seconds,
                    "samples": gpu_sampler.samples,
                },
                "artifacts": _artifact_inventory(run_dir),
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

    if not verify_run_artifacts(run_dir):
        raise RuntimeError(f"run artifact verification failed: {run_dir}")
    print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    run_experiment(
        submission_path=args.submission,
        experiment=args.experiment,
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        sample_gpu=args.sample_gpu,
        save_checkpoint=args.save_checkpoint,
    )


if __name__ == "__main__":
    cli()
