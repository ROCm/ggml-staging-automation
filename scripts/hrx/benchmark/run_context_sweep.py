#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure HRX and Vulkan prefill and decode throughput at long contexts.

Each prefill run measures ``pp N``. Each decode run primes ``N - decode_tokens``
tokens outside the timed section with llama-bench's depth option, then measures
the requested number of decode tokens. Separate invocations keep pp and tg
throughput independent; the combined ``-pg`` result is intentionally unused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_output import append_batch_log, atomic_write_json, merge_benchmark_output
from benchmark_outcomes import (
    MODEL_FAILURE_KINDS,
    aggregate_status,
    classify_outcome,
    is_unexpected,
)
from benchmark_provenance import source_revisions
from run_batched_benchmark import ModelSpec, load_manifest


DEFAULT_CONTEXTS = (32768, 65536, 131072, 262144)


class ContextSweepError(RuntimeError):
    """Raised for invalid sweep configuration or inputs."""


def parse_context_lengths(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from exc


def log(message: str) -> None:
    print(message, flush=True)


def resolve_models(manifest_path: Path, models_dir: Path, names: list[str]) -> list[tuple[ModelSpec, Path]]:
    manifest = load_manifest(manifest_path)
    by_name = {model.name: model for model in manifest.models}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ContextSweepError(f"Models are not in {manifest_path}: {unknown!r}")
    resolved = []
    for name in names:
        spec = by_name[name]
        path = models_dir / spec.name / spec.filename
        if not path.is_file():
            raise ContextSweepError(f"Model {name} is not staged at {path}")
        resolved.append((spec, path))
    return resolved


def inspect_model(evaluator: Path, model_path: Path, timeout: float) -> tuple[int | None, str | None]:
    result = subprocess.run(
        [os.fspath(evaluator), "--inspect", os.fspath(model_path)],
        text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout,
    )
    if result.returncode != 0:
        return None, result.stdout.strip() or f"metadata inspection exited {result.returncode}"
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
        return int(metadata["model_context"]), None
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, "metadata inspector returned invalid JSON"


def run_measurement(
    llama_bench: Path, model_path: Path, device: str, context_length: int,
    model_context: int, kind: str, args: argparse.Namespace, log_path: Path,
) -> dict[str, Any]:
    if context_length > model_context:
        return {
            "status": "unsupported", "reason": "requested context exceeds model training context",
            "failure_kind": "model-context", "model_context": model_context,
            "context_length": context_length,
        }
    if kind == "prefill":
        prompt, generated, depth = context_length, 0, 0
    else:
        prompt, generated, depth = 0, args.decode_tokens, context_length - args.decode_tokens
    command = [
        os.fspath(llama_bench), "-m", os.fspath(model_path), "-dev", device,
        "-ngl", str(args.gpu_layers), "-fa", args.flash_attn,
        "-p", str(prompt), "-n", str(generated), "-d", str(depth),
        "-b", str(args.batch_size), "-ub", str(args.ubatch_size),
        "-r", str(args.repetitions), "-o", "json",
    ]
    warmup_command = list(command)
    repetitions_index = warmup_command.index("-r") + 1
    warmup_command[repetitions_index] = "1"
    warmup_command.insert(-2, "--no-warmup")

    def execute(argv: list[str]) -> tuple[str, str, int | None, bool, float]:
        started = time.monotonic()
        try:
            result = subprocess.run(
                argv, text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=args.run_timeout_seconds,
            )
            return result.stdout, result.stderr, result.returncode, False, time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return stdout, stderr, None, True, time.monotonic() - started

    warmup_stdout, warmup_stderr, warmup_exit, warmup_timeout, warmup_duration = execute(warmup_command)
    if warmup_exit == 0 and not warmup_timeout:
        log("++ " + shlex.join(command))
        stdout, stderr, exit_code, timed_out, duration_s = execute(command)
    else:
        stdout, stderr = warmup_stdout, warmup_stderr
        exit_code, timed_out, duration_s = warmup_exit, warmup_timeout, 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== {device} {kind} context={context_length} =====\n")
        handle.write("===== exact-workload warmup (not measured) =====\n")
        handle.write("++ " + shlex.join(warmup_command) + "\n")
        handle.write(f"exit={warmup_exit} duration_s={warmup_duration:.3f}\n")
        handle.write("--- stdout ---\n" + warmup_stdout + ("" if warmup_stdout.endswith("\n") else "\n"))
        handle.write("--- stderr ---\n" + warmup_stderr + ("" if warmup_stderr.endswith("\n") else "\n"))
        if warmup_exit == 0 and not warmup_timeout:
            handle.write("===== measured repetitions =====\n")
            handle.write("++ " + shlex.join(command) + "\n")
            handle.write(f"exit={exit_code} duration_s={duration_s:.3f}\n")
            handle.write("--- stdout ---\n" + stdout + ("" if stdout.endswith("\n") else "\n"))
            handle.write("--- stderr ---\n" + stderr + ("" if stderr.endswith("\n") else "\n"))

    measurement: dict[str, Any] = {
        "status": "failed", "error": None, "failure_kind": None,
        "exit_code": exit_code, "duration_s": round(duration_s, 3),
        "log": log_path.name, "command": command,
        "warmup_command": warmup_command, "warmup_exit_code": warmup_exit,
        "warmup_duration_s": round(warmup_duration, 3),
        "context_length": context_length, "model_context": model_context,
        "prompt_tokens": prompt, "decode_tokens": generated,
        "decode_history_tokens": depth,
        "llama_bench_default_warmup": True,
        "explicit_exact_workload_warmups": 1,
        "repetitions": args.repetitions,
    }
    if not timed_out and exit_code == 0:
        try:
            rows = json.loads(stdout)
            if not isinstance(rows, list) or len(rows) != 1:
                raise ValueError("expected one llama-bench JSON row")
            row = rows[0]
            expected_prompt = prompt
            expected_generated = generated
            expected_depth = depth
            if (row.get("n_prompt") != expected_prompt or row.get("n_gen") != expected_generated
                    or row.get("n_depth") != expected_depth):
                raise ValueError("llama-bench returned a row with different token counts")
            throughput = float(row["avg_ts"])
            if not throughput > 0:
                raise ValueError("llama-bench returned nonpositive throughput")
            measurement.update({
                "status": "ok", "tokens_per_second": throughput,
                "stddev_tokens_per_second": float(row["stddev_ts"]),
                "avg_ns": int(row["avg_ns"]), "stddev_ns": int(row["stddev_ns"]),
                "backend": row.get("backends"), "gpu_layers": row.get("n_gpu_layers"),
                "cache_type_k": row.get("type_k"), "cache_type_v": row.get("type_v"),
                "llama_bench_row": row,
            })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            measurement.update({
                "error": f"invalid llama-bench result: {exc}",
                "failure_kind": "invalid-measurement",
            })
    else:
        combined = (stdout + "\n" + stderr).lower()
        if timed_out:
            error, failure_kind = f"timed out after {args.run_timeout_seconds} seconds", "execution"
        elif exit_code == -9 or any(term in combined for term in ("out of memory", "allocation failed", "failed to allocate", "out of device memory", "std::bad_alloc")):
            error, failure_kind = f"llama-bench exited {exit_code} after allocation failure", "resource-exhausted"
        elif any(term in combined for term in ("no matching device", "device not found", "unknown device", "failed to initialize backend", "unsupported device")):
            error, failure_kind = f"backend device unavailable: {device}", "infrastructure"
        elif any(term in combined for term in (
            "prefill failed", "decode failed", "failed to decode",
            "llama_decode failed", "failed to evaluate", "evaluation failed",
            "failed to compute", "graph compute failed", "nonfinite", "non-finite",
        )):
            error, failure_kind = f"model execution failed: llama-bench exited {exit_code}", "model-execution"
        else:
            error, failure_kind = f"llama-bench exited with status {exit_code}", "infrastructure"
        measurement.update({"error": error, "failure_kind": failure_kind})
    return measurement


def run(args: argparse.Namespace) -> int:
    llama_bench = args.llama_bench.resolve()
    evaluator = args.context_inspector.resolve()
    models_dir = args.models_dir.resolve()
    models = resolve_models(args.model_manifest, models_dir, args.models)
    if (not args.context_lengths or args.decode_tokens < 1
            or any(length <= args.decode_tokens for length in args.context_lengths)):
        raise ContextSweepError("every context length must exceed --decode-tokens")
    for binary in (llama_bench, evaluator):
        if not binary.is_file():
            raise ContextSweepError(f"Required benchmark binary is missing: {binary}")

    logs = {"hrx": args.hrx_log, "vulkan": args.vulkan_log}
    for path in logs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not args.batched:
            path.write_text("", encoding="utf-8")

    document: dict[str, Any] = {
        "schema_version": 1,
        "source_revisions": source_revisions(),
        "llama_bench": os.fspath(llama_bench),
        "context_inspector": os.fspath(evaluator),
        "devices": {"hrx": args.hrx_device, "vulkan": args.vulkan_device},
        "settings": {
            "context_lengths": list(args.context_lengths),
            "prefill_tokens": "context length N measured independently",
            "decode_tokens": args.decode_tokens,
            "decode_history_tokens": "N - decode_tokens, populated before timer",
            "warmups": "one explicit exact-workload warmup, plus llama-bench default warmup before measured repetitions",
            "repetitions": args.repetitions,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "gpu_layers": args.gpu_layers,
            "flash_attention": args.flash_attn,
        },
        "models": [],
    }
    unexpected = False
    for spec, model_path in models:
        model_context, inspect_error = inspect_model(evaluator, model_path, args.run_timeout_seconds)
        row: dict[str, Any] = {
            "model": spec.name, "model_id": spec.id, "file": spec.filename,
            "model_sha256": spec.sha256, "model_revision": spec.revision,
            "model_context": model_context, "batch": args.batch_number, "contexts": {},
            "expected_results": {backend: {} for backend in ("hrx", "vulkan")},
        }
        if inspect_error:
            for context_length in args.context_lengths:
                by_backend = {}
                for backend in logs:
                    expectation = spec.expectation("context-sweep", backend, str(context_length))
                    row["expected_results"][backend][str(context_length)] = expectation
                    measurement = {
                        "status": "failed", "error": inspect_error,
                        "failure_kind": "infrastructure", "log": logs[backend].name,
                    }
                    outcome = classify_outcome(
                        "fail", expectation, expected_failure_allowed=False,
                        infrastructure_failure=True,
                    )
                    by_backend[backend] = {
                        "prefill": dict(measurement), "decode": dict(measurement),
                        "result": "fail", "outcome": outcome,
                    }
                    unexpected = unexpected or is_unexpected(outcome)
                row["contexts"][str(context_length)] = by_backend
            for backend in logs:
                row["expected_results"][backend] = {
                    str(length): spec.expectation("context-sweep", backend, str(length))
                    for length in args.context_lengths
                }
        else:
            for context_length in args.context_lengths:
                by_backend = {}
                for backend, device in (("hrx", args.hrx_device), ("vulkan", args.vulkan_device)):
                    active_log = logs[backend]
                    if args.batched:
                        active_log = models_dir.parent / f"context-sweep-{backend}.log"
                    by_backend[backend] = {
                        "prefill": run_measurement(
                            llama_bench, model_path, device, context_length, model_context,
                            "prefill", args, active_log,
                        ),
                        "decode": run_measurement(
                            llama_bench, model_path, device, context_length, model_context,
                            "decode", args, active_log,
                        ),
                    }
                row["contexts"][str(context_length)] = by_backend
                for backend, measurements in list(by_backend.items()):
                    expectation = spec.expectation("context-sweep", backend, str(context_length))
                    row["expected_results"][backend][str(context_length)] = expectation
                    raw_result = aggregate_status(
                        measurement["status"] for measurement in measurements.values()
                    )
                    failed_measurements = [
                        measurement for measurement in measurements.values()
                        if measurement["status"] != "ok"
                    ]
                    failure_kinds = [
                        measurement.get("failure_kind") for measurement in failed_measurements
                    ]
                    expected_failure_allowed = all(
                        kind in MODEL_FAILURE_KINDS for kind in failure_kinds
                    )
                    infrastructure_failure = any(
                        kind not in MODEL_FAILURE_KINDS for kind in failure_kinds
                    )
                    outcome = classify_outcome(
                        raw_result, expectation,
                        expected_failure_allowed=expected_failure_allowed,
                        infrastructure_failure=infrastructure_failure,
                    )
                    measurements["result"] = raw_result
                    measurements["outcome"] = outcome
                    unexpected = unexpected or is_unexpected(outcome)
                log(f"{spec.id} context={context_length}: HRX pp={by_backend['hrx']['prefill'].get('tokens_per_second')} tg={by_backend['hrx']['decode'].get('tokens_per_second')}; Vulkan pp={by_backend['vulkan']['prefill'].get('tokens_per_second')} tg={by_backend['vulkan']['decode'].get('tokens_per_second')}")
        document["models"].append(row)
        row["result"] = aggregate_status(
            values["hrx"]["result"] for values in row["contexts"].values()
        )
        row["reference_result"] = aggregate_status(
            values["vulkan"]["result"] for values in row["contexts"].values()
        )
        hrx_outcomes = [values["hrx"]["outcome"] for values in row["contexts"].values()]
        vulkan_outcomes = [values["vulkan"]["outcome"] for values in row["contexts"].values()]
        row["outcome"] = "FAIL" if "FAIL" in hrx_outcomes else (
            "XPASS" if "XPASS" in hrx_outcomes else "XFAIL" if "XFAIL" in hrx_outcomes else "PASS"
        )
        row["reference_outcome"] = "FAIL" if "FAIL" in vulkan_outcomes else (
            "XPASS" if "XPASS" in vulkan_outcomes else "XFAIL" if "XFAIL" in vulkan_outcomes else "PASS"
        )

    if args.batched:
        if args.batch_number is None:
            raise ContextSweepError("--batch-number is required with --batched")
        merge_benchmark_output(args.output, document)
        for backend, log_path in logs.items():
            source = models_dir.parent / f"context-sweep-{backend}.log"
            try:
                append_batch_log(log_path, source, args.batch_number)
            except OSError as exc:
                log(f"Warning: could not append {backend} batch log: {exc}")
    else:
        atomic_write_json(args.output, document)
    log(f"Wrote {args.output}")
    return 1 if unexpected else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--context-inspector", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hrx-log", type=Path, required=True)
    parser.add_argument("--vulkan-log", type=Path, required=True)
    parser.add_argument("--hrx-device", default="HRX0")
    parser.add_argument("--vulkan-device", default="Vulkan0")
    parser.add_argument("--contexts", dest="context_lengths", type=parse_context_lengths, default=DEFAULT_CONTEXTS)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.batched and args.batch_number is None:
        parser.error("--batch-number is required with --batched")
    if args.batch_number is not None and not args.batched:
        parser.error("--batch-number requires --batched")
    if (not args.context_lengths or tuple(sorted(set(args.context_lengths))) != args.context_lengths
            or any(length <= 0 for length in args.context_lengths)):
        parser.error("--contexts must be unique positive integers in increasing order")
    if args.decode_tokens < 1 or any(length <= args.decode_tokens for length in args.context_lengths):
        parser.error("--decode-tokens must be positive and less than every context")
    if args.batch_size < 1 or args.ubatch_size < 1:
        parser.error("--batch-size and --ubatch-size must be positive")
    if not math.isfinite(args.run_timeout_seconds) or args.run_timeout_seconds <= 0:
        parser.error("--run-timeout-seconds must be finite and positive")
    return args


def main() -> int:
    try:
        return run(parse_args())
    except (ContextSweepError, OSError, subprocess.SubprocessError) as exc:
        print(f"Context sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
