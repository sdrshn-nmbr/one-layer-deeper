"""Minimal Modal runner for one-layer-benchmark experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import modal

from service.tiers import submission_manifest_timeouts


APP_NAME = "one-layer-benchmark-runner"
REMOTE_ROOT = "/workspace/one-layer-benchmark"
ARTIFACT_ROOT = "/artifacts"
ARTIFACT_VOLUME_NAME = "one-layer-deeper-artifacts"
SUBMISSION_FUNCTION_NAME = "evaluate_submission"
SUBMISSION_MANIFEST_TIMEOUTS = submission_manifest_timeouts()

SMOKE_MANIFEST_FILENAME = "h100_easy_e1.json"
SMOKE_SUBMISSION_FILE = "submissions/baseline_adamw/submission.py"
SMOKE_WAIT_GRACE_SECONDS = 120

DEFAULT_TIMEOUT_SECONDS = 45 * 60
MAX_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_GPU = "H100"
GPU_CONFIGS = {
    "A100": ("run_a100", "A100-80GB"),
    "H100": ("run_h100", "H100"),
    "B200": ("run_b200", "B200"),
}
GPU_FUNCTIONS = {gpu: function_name for gpu, (function_name, _) in GPU_CONFIGS.items()}


app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "build-essential")
    .pip_install("uv==0.11.3")
    .env({"PYTHONPATH": REMOTE_ROOT})
    .run_commands(f"mkdir -p {REMOTE_ROOT}")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .add_local_file("submission_validation.py", f"{REMOTE_ROOT}/submission_validation.py", copy=True)
    .run_commands(
        f"cd {REMOTE_ROOT} && uv sync --frozen --no-install-project",
    )
    .add_local_dir(
        "benchmark",
        f"{REMOTE_ROOT}/benchmark",
        copy=True,
        ignore=["__pycache__"],
    )
    .add_local_file("data/__init__.py", f"{REMOTE_ROOT}/data/__init__.py", copy=True)
    .add_local_file(
        "service/__init__.py",
        f"{REMOTE_ROOT}/service/__init__.py",
        copy=True,
    )
    .add_local_file(
        "service/tiers.py",
        f"{REMOTE_ROOT}/service/tiers.py",
        copy=True,
    )
    .add_local_file("data/config.py", f"{REMOTE_ROOT}/data/config.py", copy=True)
    .add_local_file("data/factory.py", f"{REMOTE_ROOT}/data/factory.py", copy=True)
    .add_local_file("data/counting.py", f"{REMOTE_ROOT}/data/counting.py", copy=True)
    .add_local_file("data/squaring_mod.py", f"{REMOTE_ROOT}/data/squaring_mod.py", copy=True)
    .add_local_file(
        "scripts/generate_datasets.sh",
        f"{REMOTE_ROOT}/scripts/generate_datasets.sh",
        copy=True,
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && uv run --no-sync bash scripts/generate_datasets.sh",
    )
    .add_local_dir(
        "research",
        f"{REMOTE_ROOT}/research",
        copy=True,
        ignore=["__pycache__"],
    )
    .add_local_dir(
        "submissions/recurrent",
        f"{REMOTE_ROOT}/submissions/recurrent",
        copy=True,
        ignore=["__pycache__"],
    )
    .add_local_python_source("modal_runner")
)


def _normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing command; pass it after --")
    if command[0].endswith(".py"):
        return ["python", *command]
    return command


def _validate_timeout(timeout_seconds: int) -> int:
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds cannot exceed {MAX_TIMEOUT_SECONDS}")
    return timeout_seconds


def _remote_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{REMOTE_ROOT}/.venv/bin:{env.get('PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
        }
    )
    return env


def _run_command(command: list[str], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    command = _normalize_command(command)
    timeout_seconds = _validate_timeout(timeout_seconds)
    started = time.monotonic()
    log_tail: deque[str] = deque(maxlen=400)

    print(f"[modal-runner] cwd={REMOTE_ROOT}", flush=True)
    print(f"[modal-runner] command={shlex.join(command)}", flush=True)
    print(f"[modal-runner] timeout_seconds={timeout_seconds}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env=_remote_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    timed_out = False
    assert process.stdout is not None
    output: Queue[str | None] = Queue()

    def read_output() -> None:
        for item in process.stdout:
            output.put(item)
        output.put(None)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    stream_closed = False
    while True:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0 and process.poll() is None:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        try:
            line = output.get(timeout=max(0.01, min(0.2, max(remaining, 0.01))))
        except Empty:
            line = ""
        if line is None:
            stream_closed = True
        elif line:
            print(line, end="", flush=True)
            log_tail.append(line)
        if process.poll() is not None and stream_closed:
            break

    reader.join(timeout=1)

    process.stdout.close()
    return {
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "duration_seconds": time.monotonic() - started,
        "log_tail": "".join(log_tail),
    }


def _register_gpu_function(name: str, gpu: str) -> modal.Function:
    @app.function(
        gpu=gpu,
        image=image,
        timeout=MAX_TIMEOUT_SECONDS,
        serialized=True,
        name=name,
        volumes={ARTIFACT_ROOT: artifact_volume},
    )
    def run(command: list[str], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
        result = _run_command(command, timeout_seconds)
        artifact_volume.commit()
        return result

    return run


run_a100 = _register_gpu_function(*GPU_CONFIGS["A100"])
run_h100 = _register_gpu_function(*GPU_CONFIGS["H100"])
run_b200 = _register_gpu_function(*GPU_CONFIGS["B200"])


def _parse_benchmark_result(log_tail: str) -> dict[str, Any]:
    for line in reversed(log_tail.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise ValueError("benchmark process returned no RESULT_JSON record")


@app.function(
    gpu="H100",
    image=image,
    timeout=MAX_TIMEOUT_SECONDS,
    block_network=True,
    serialized=True,
    name=SUBMISSION_FUNCTION_NAME,
)
def evaluate_submission(
    submission_source: str,
    manifest_filename: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Evaluate one uploaded source file in a disposable H100 container."""

    expected_timeout = SUBMISSION_MANIFEST_TIMEOUTS.get(manifest_filename)
    if expected_timeout is None:
        raise ValueError(f"manifest is not available for submissions: {manifest_filename}")
    if timeout_seconds != expected_timeout:
        raise ValueError("timeout does not match the selected submission manifest")
    encoded = submission_source.encode("utf-8")
    if not encoded or len(encoded) > 256 * 1024:
        raise ValueError("submission must contain 1 to 262144 UTF-8 bytes")

    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            delete=False,
        ) as file:
            file.write(submission_source)
            path = file.name
        result = _run_command(
            [
                "python",
                "-m",
                "benchmark.runner",
                "--manifest",
                f"benchmark/manifests/{manifest_filename}",
                "--submission-file",
                path,
                "--include-structured-metrics",
            ],
            timeout_seconds=timeout_seconds,
        )
        if result["returncode"] == 0 and not result["timed_out"]:
            result["benchmark_result"] = _parse_benchmark_result(result["log_tail"])
        return result
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _smoke_failure(message: str, result: dict[str, Any]) -> RuntimeError:
    log_tail = result.get("log_tail")
    if isinstance(log_tail, str) and log_tail:
        message = f"{message}\n--- remote log tail ---\n{log_tail}"
    return RuntimeError(message)


def _validate_smoke_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise RuntimeError("Modal smoke test returned a non-object result")
    if result.get("timed_out"):
        raise _smoke_failure("Modal smoke evaluation timed out", result)
    if result.get("returncode") != 0:
        raise _smoke_failure(
            f"Modal smoke evaluator exited with code {result.get('returncode')!r}",
            result,
        )

    benchmark_result = result.get("benchmark_result")
    if not isinstance(benchmark_result, dict):
        raise _smoke_failure("Modal smoke test returned no benchmark_result", result)
    score = (benchmark_result.get("score") or {}).get("mean_exact_accuracy")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise _smoke_failure(
            f"Modal smoke test returned an invalid mean_exact_accuracy: {score!r}",
            result,
        )
    return benchmark_result


def smoke(
    submission_file: str = SMOKE_SUBMISSION_FILE,
    manifest_filename: str = SMOKE_MANIFEST_FILENAME,
) -> None:
    expected_timeout = SUBMISSION_MANIFEST_TIMEOUTS.get(manifest_filename)
    if expected_timeout is None:
        raise ValueError(f"unknown smoke manifest: {manifest_filename}")
    source_path = Path(submission_file)
    submission_source = source_path.read_text(encoding="utf-8")

    print(f"[modal-smoke] app={APP_NAME}", flush=True)
    print(f"[modal-smoke] function={SUBMISSION_FUNCTION_NAME}", flush=True)
    print(f"[modal-smoke] submission={source_path}", flush=True)
    print(f"[modal-smoke] manifest={manifest_filename}", flush=True)
    function = modal.Function.from_name(APP_NAME, SUBMISSION_FUNCTION_NAME)
    call = function.spawn(
        submission_source=submission_source,
        manifest_filename=manifest_filename,
        timeout_seconds=expected_timeout,
    )
    print(f"[modal-smoke] call_id={call.object_id}", flush=True)
    try:
        result = call.get(timeout=expected_timeout + SMOKE_WAIT_GRACE_SECONDS)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Modal smoke call {call.object_id} did not finish within "
            f"{expected_timeout + SMOKE_WAIT_GRACE_SECONDS} seconds"
        ) from exc

    benchmark_result = _validate_smoke_result(result)
    print(
        "SMOKE_RESULT_JSON="
        + json.dumps(
            {"call_id": call.object_id, "benchmark_result": benchmark_result},
            sort_keys=True,
        ),
        flush=True,
    )


def submit(command: list[str], gpu: str, timeout_seconds: int) -> None:
    gpu = gpu.upper()
    if gpu not in GPU_FUNCTIONS:
        raise ValueError(f"unsupported GPU {gpu!r}; choose one of {', '.join(GPU_FUNCTIONS)}")

    function = modal.Function.from_name(APP_NAME, GPU_FUNCTIONS[gpu])
    call = function.spawn(command=_normalize_command(command), timeout_seconds=_validate_timeout(timeout_seconds))
    print(call.object_id)


def wait(call_id: str, timeout: float | None) -> None:
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=timeout)
    except TimeoutError:
        print(f"{call_id}: still running", file=sys.stderr)
        raise SystemExit(124)

    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("timed_out"):
        raise SystemExit(124)
    raise SystemExit(int(result.get("returncode") or 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    submit_parser = subparsers.add_parser("submit", help="submit one Modal job")
    submit_parser.add_argument("--gpu", choices=sorted(GPU_FUNCTIONS), default=DEFAULT_GPU)
    submit_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    submit_parser.add_argument("command", nargs=argparse.REMAINDER)

    wait_parser = subparsers.add_parser("wait", help="wait for one Modal call")
    wait_parser.add_argument("call_id")
    wait_parser.add_argument("--timeout", type=float, default=None)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="run a baseline submission against the deployed evaluator",
    )
    smoke_parser.add_argument("--submission-file", default=SMOKE_SUBMISSION_FILE)
    smoke_parser.add_argument("--manifest-filename", default=SMOKE_MANIFEST_FILENAME)

    args = parser.parse_args()
    if args.subcommand == "submit":
        submit(args.command, args.gpu, args.timeout_seconds)
    elif args.subcommand == "wait":
        wait(args.call_id, args.timeout)
    elif args.subcommand == "smoke":
        smoke(args.submission_file, args.manifest_filename)


if __name__ == "__main__":
    main()
