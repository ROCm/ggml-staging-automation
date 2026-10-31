#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check pinned video and image questions through the CI-built llama-server.

The batch driver supplies checksum-verified text and projector GGUFs. Each
backend gets a fresh server with both files fully placed on that backend's GPU.
Every fixture is sent once for warmup and three times for measurement. Answers
gate correctness; llama-server's timing fields remain diagnostic measurements.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_output import atomic_write_json, merge_benchmark_output
from run_batched_benchmark import ModelSpec, load_manifest

WARMUP_RUNS = 1
MEASURED_RUNS = 3
REQUEST_TIMEOUT_SECONDS = 600
READY_TIMEOUT_SECONDS = 600
BACKEND_DEVICES = {"hrx": "HRX0", "vulkan": "Vulkan0"}


@dataclass(frozen=True)
class VideoCase:
    id: str
    kind: str
    path: Path
    question: str
    answer: str


def load_cases(path: Path) -> list[VideoCase]:
    """Load the fixture contract at the CLI boundary."""
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in document["cases"]:
        kind = entry["type"]
        if kind not in {"image", "video"}:
            raise ValueError(f"Unsupported fixture type: {kind}")
        media_path = path.parent / entry["file"]
        if not media_path.is_file():
            raise ValueError(f"Missing fixture: {media_path}")
        cases.append(
            VideoCase(entry["id"], kind, media_path, entry["question"], entry["answer"])
        )
    if not cases:
        raise ValueError("Video test set has no cases")
    return cases


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request(
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        value = json.loads(response.read().decode("utf-8"))
        return response.status, value
    finally:
        connection.close()


def wait_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited {process.returncode} before readiness"
            )
        try:
            status, _ = request(port, "GET", "/health", timeout=2.0)
            if status == 200:
                return
            last_error = f"GET /health returned {status}"
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(
        f"llama-server was not ready in {READY_TIMEOUT_SECONDS}s: {last_error}"
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def append_server_log(
    destination: Path, source: Path, model_id: str, backend: str, batch_number: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = f"batch {batch_number}, {model_id}, {backend}"
    with destination.open("a", encoding="utf-8") as output:
        output.write(f"===== BEGIN {label} =====\n")
        if source.is_file():
            with source.open(encoding="utf-8", errors="replace") as input_file:
                shutil.copyfileobj(input_file, output)
        output.write(f"\n===== END {label} =====\n")


def media_content(case: VideoCase) -> dict[str, Any]:
    encoded = base64.b64encode(case.path.read_bytes()).decode("ascii")
    if case.kind == "video":
        return {"type": "input_video", "input_video": {"data": encoded}}
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def parse_choice(content: str) -> str | None:
    match = re.fullmatch(r"\s*([A-Z])[.)]?\s*", content.upper())
    return match.group(1) if match else None


def run_sample(
    port: int, case: VideoCase, media: dict[str, Any], phase: str, index: int
) -> dict[str, Any]:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": case.question}, media],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
        "cache_prompt": False,
    }
    started = time.monotonic()
    sample: dict[str, Any] = {"phase": phase, "index": index}
    try:
        status, document = request(
            port, "POST", "/v1/chat/completions", body, timeout=REQUEST_TIMEOUT_SECONDS
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {document}")
        content = document["choices"][0]["message"]["content"]
        timings = document["timings"]
        choice = parse_choice(content)
        sample.update(
            answer=content,
            choice=choice,
            correct=choice == case.answer,
            prompt_ms=timings["prompt_ms"],
            predicted_ms=timings["predicted_ms"],
            predicted_tps=timings["predicted_per_second"],
        )
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        http.client.HTTPException,
        RuntimeError,
    ) as exc:
        sample.update(correct=False, error=str(exc))
    sample["wall_ms"] = (time.monotonic() - started) * 1000
    return sample


def run_backend(
    args: argparse.Namespace, model: ModelSpec, backend: str, cases: list[VideoCase]
) -> dict[str, Any]:
    device = BACKEND_DEVICES[backend]
    model_dir = args.models_dir.resolve() / model.name
    text_path = model_dir / model.filename
    projector_path = model_dir / model.projector.filename
    port = free_port()
    command = [
        str(args.llama_server.resolve()),
        "--model",
        str(text_path),
        "--mmproj",
        str(projector_path),
        "--device",
        device,
        "--mmproj-device",
        device,
        "--n-gpu-layers",
        "all",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--no-cache-prompt",
        "--video-fps",
        "4",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print("++ " + " ".join(command), flush=True)
    result: dict[str, Any] = {"device": device, "command": command, "cases": []}
    with tempfile.TemporaryDirectory(prefix="video-benchmark-") as temporary_dir:
        server_log = Path(temporary_dir) / "server.log"
        process = None
        try:
            with server_log.open("wb") as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            wait_ready(process, port)
            for case in cases:
                media = media_content(case)
                samples = [
                    run_sample(
                        port,
                        case,
                        media,
                        "warmup" if run_index < WARMUP_RUNS else "measured",
                        run_index,
                    )
                    for run_index in range(WARMUP_RUNS + MEASURED_RUNS)
                ]
                passed = all(sample["correct"] for sample in samples)
                result["cases"].append(
                    {
                        "id": case.id,
                        "type": case.kind,
                        "expected": case.answer,
                        "passed": passed,
                        "samples": samples,
                    }
                )
                print(
                    f"{'PASS' if passed else 'FAIL'}: {model.id} {backend} {case.id}",
                    flush=True,
                )
        except (OSError, RuntimeError) as exc:
            result["error"] = str(exc)
        finally:
            if process is not None:
                stop_server(process)
            append_server_log(
                getattr(args, backend + "_log"),
                server_log,
                model.id,
                backend,
                args.batch_number or 1,
            )
    server_succeeded = "error" not in result
    all_cases_ran = len(result["cases"]) == len(cases)
    all_answers_correct = all(case["passed"] for case in result["cases"])
    result["passed"] = server_succeeded and all_cases_ran and all_answers_correct
    return result


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.model_manifest)
    by_name = {model.name: model for model in manifest.models}
    cases = load_cases(args.test_set)
    rows = []
    for name in args.models:
        model = by_name[name]
        supports_video = "video" in model.capabilities
        has_projector = model.projector is not None
        ready_for_video = supports_video and has_projector
        if not ready_for_video:
            raise ValueError(f"Model {model.id} is not video-enabled")
        backends = {
            backend: run_backend(args, model, backend, cases)
            for backend in BACKEND_DEVICES
        }
        passed = all(result["passed"] for result in backends.values())
        rows.append(
            {
                "model_id": model.id,
                "model": model.name,
                "weights": {"revision": model.revision, "sha256": model.sha256},
                "projector": {
                    "filename": model.projector.filename,
                    "sha256": model.projector.sha256,
                },
                "backends": backends,
                "passed": passed,
                "batch": args.batch_number,
            }
        )
    document = {
        "test_set": args.test_set.name,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "models": rows,
    }
    if args.batched:
        merge_benchmark_output(args.output, document)
    else:
        atomic_write_json(args.output, document)
    return 0 if all(row["passed"] for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "llama-server",
        "model-manifest",
        "test-set",
        "models-dir",
        "output",
        "hrx-log",
        "vulkan-log",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Video benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
