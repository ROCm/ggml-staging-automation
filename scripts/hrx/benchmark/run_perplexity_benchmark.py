#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure two perplexity regimes on HRX and Vulkan over the same pinned corpus.

Four sequential phases measure HRX prefill-like, Vulkan prefill-like, HRX
decode-like, then Vulkan decode-like. Prefill-like uses a 512-token microbatch;
decode-like uses a single token. Every phase measures all models in its batch.

One JSON artifact contains complete model rows: each regime holds its HRX and
Vulkan measurements, ratio, and numerical verdict. Execution and invalid-estimate
failures remain distinct; only valid pairs receive a numerical verdict. Batch
merging simply appends complete rows to models using the shared output helper.

Each row's result and outcome aggregate HRX across both regimes before applying
its one perplexity expectation. SKIP still collects measurements and preserves
the raw result. Vulkan failures are always fatal and recorded in reference_result.
The combined artifact and separate backend logs are written before returning a
failing exit status. The report consumes this artifact as one comparison table.
Lemonade benchmark expectations are separate.

Invocation, input, writing, and cleanup errors retain their fatal exception path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from benchmark_output import (
    append_batch_log,
    atomic_write_json,
    merge_benchmark_output,
)
from run_batched_benchmark import ModelSpec, load_manifest


FINAL_ESTIMATE_PATTERN = re.compile(
    r"Final estimate: PPL = (?P<value>\S+) \+/- (?P<uncertainty>\S+)"
)


REGIMES = {
    "prefill-like": {"name": "Prefill like", "ctx": 512, "batch": 512,
                     "microbatch": 512, "chunks": 32},
    "decode-like": {"name": "Decode like", "ctx": 512, "batch": 512,
                    "microbatch": 1, "chunks": 2},
}


def numerical_verdict(hrx: dict, vulkan: dict, maximum: float) -> str:
    """Execution/estimate failures cannot count as numerical detections."""
    hrx_valid = hrx["status"] == "ok"
    vulkan_valid = vulkan["status"] == "ok"
    comparable = hrx_valid and vulkan_valid
    if not comparable:
        return "unavailable"
    exceeds_limit = hrx["ppl"]["value"] > maximum * vulkan["ppl"]["value"]
    return "fail" if exceeds_limit else "pass"


class PerplexityBenchmarkError(RuntimeError):
    """Raised when the perplexity benchmark cannot run as requested."""


@dataclass
class PerplexityPhase:
    name: str
    backend: str
    device: str
    regime_id: str
    log: Path


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    path: Path


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models(
    manifest_path: Path, models_dir: Path, names: list[str]
) -> list[ResolvedModel]:
    """Map Lemonade model names to the staged GGUF files they refer to."""
    manifest = load_manifest(manifest_path)
    specs_by_name = {spec.name: spec for spec in manifest.models}
    unknown_names = [name for name in names if name not in specs_by_name]
    if unknown_names:
        raise PerplexityBenchmarkError(
            f"Models are not in {manifest_path}: {unknown_names!r}"
        )
    resolved: list[ResolvedModel] = []
    for name in names:
        spec = specs_by_name[name]
        path = models_dir / spec.directory / spec.filename
        if not path.is_file():
            raise PerplexityBenchmarkError(f"Model {name} is not staged at {path}")
        resolved.append(ResolvedModel(spec=spec, path=path))
    return resolved


def log_llama_perplexity_devices(llama_perplexity: Path) -> None:
    """Record the devices llama-perplexity enumerates before measuring."""
    result = subprocess.run(
        [os.fspath(llama_perplexity), "--list-devices"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(f"llama-perplexity --list-devices (exit {result.returncode}):")
    log(result.stdout.rstrip())


def parse_final_estimate(text: str) -> dict[str, float] | None:
    match = FINAL_ESTIMATE_PATTERN.search(text)
    if match is None:
        return None
    try:
        value = float(match.group("value"))
        uncertainty = float(match.group("uncertainty"))
    except ValueError:
        return None
    value_valid = math.isfinite(value) and value > 0
    uncertainty_valid = math.isfinite(uncertainty) and uncertainty >= 0
    estimate_valid = value_valid and uncertainty_valid
    if not estimate_valid:
        return None
    return {"value": value, "uncertainty": uncertainty}


def run_perplexity(
    llama_perplexity: Path,
    model: ResolvedModel,
    corpus_file: Path,
    phase: PerplexityPhase,
    args: argparse.Namespace,
    log_handle: TextIO,
) -> tuple[dict[str, Any], bool]:
    """Measure one model on one device and record the outcome as a row."""
    regime_id = phase.regime_id
    regime = REGIMES[regime_id]
    command = [
        os.fspath(llama_perplexity),
        *args.perplexity_arg,
        "-m",
        os.fspath(model.path),
        "-f",
        os.fspath(corpus_file),
        "-c",
        str(regime["ctx"]),
        "--chunks",
        str(regime["chunks"]),
        "-b",
        str(regime["batch"]),
        "-ub",
        str(regime["microbatch"]),
        "--device",
        phase.device,
    ]
    log("++ " + shlex.join(command))
    log_handle.write(f"===== {model.spec.name} [{regime_id}] on {phase.device} =====\n")
    log_handle.write("++ " + shlex.join(command) + "\n")
    log_handle.flush()

    started = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.run_timeout_seconds,
        )
        output = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        timed_out = True
    duration_s = time.monotonic() - started
    log_handle.write(output)
    if not output.endswith("\n"):
        log_handle.write("\n")
    log_handle.flush()

    ppl = parse_final_estimate(output)
    exited_with_error = exit_code != 0
    missing_estimate = ppl is None
    failure_kind = None
    if timed_out:
        failure_kind = "execution"
        error: str | None = f"timed out after {args.run_timeout_seconds} seconds"
    elif exited_with_error:
        failure_kind = "execution"
        error = f"llama-perplexity exited with status {exit_code}"
    elif missing_estimate:
        failure_kind = "invalid-measurement"
        error = "llama-perplexity did not print a valid finite positive PPL estimate"
    else:
        error = None
    succeeded = error is None

    summary = f"{model.spec.name} [{regime_id}] on {phase.device}: "
    if succeeded:
        assert ppl is not None
        summary += f"PPL = {ppl['value']} +/- {ppl['uncertainty']}"
    else:
        summary += f"failed: {error}"
    log(f"{summary} in {duration_s:.1f}s")

    row = {
        "status": "ok" if succeeded else "failed",
        "error": error,
        "failure_kind": failure_kind,
        "exit_code": exit_code,
        "duration_s": round(duration_s, 3),
        "ppl": ppl if succeeded else None,
        "log": phase.log.name,
        "command": command,
    }
    return row, succeeded


def run_phase(
    phase: PerplexityPhase,
    active_log: Path,
    models: list[ResolvedModel],
    llama_perplexity: Path,
    corpus_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collect one backend/regime pair without stopping on a failed model."""
    measurements = {}
    active_log.parent.mkdir(parents=True, exist_ok=True)
    with active_log.open("a", encoding="utf-8") as log_handle:
        for model in models:
            measurement, _ = run_perplexity(
                llama_perplexity, model, corpus_file, phase, args, log_handle,
            )
            measurements[model.spec.name] = measurement
    return measurements


def evaluate_models(document: dict, xfail: set[str], skip: set[str]) -> bool:
    """Apply the HRX expectation once; never waive a broken Vulkan reference."""
    unexpected = False
    maximum = document["settings"]["max_perplexity_ratio"]
    for row in document["models"]:
        name = row["model"]
        hrx_failed = False
        vulkan_failed = False
        for regime_id, pair in row["regimes"].items():
            measurement = pair["hrx"]
            baseline = pair["vulkan"]
            verdict = numerical_verdict(measurement, baseline, maximum)
            pair["numerical_verdict"] = verdict
            pair["ratio"] = (
                measurement["ppl"]["value"] / baseline["ppl"]["value"]
                if verdict != "unavailable" else None
            )
            measurement_failed = measurement["status"] != "ok"
            numerically_failed = verdict == "fail"
            hrx_failed = hrx_failed or measurement_failed or numerically_failed
            if baseline["status"] != "ok":
                log(f"FAIL: Vulkan perplexity for {name} [{regime_id}]: {baseline['error']}")
                unexpected = True
                vulkan_failed = True
            log(f"HRX perplexity for {name} [{regime_id}]: "
                f"measurement={measurement['status']}, numerical={verdict}, "
                f"ratio={pair['ratio']}")
        row["result"] = "fail" if hrx_failed else "pass"
        row["reference_result"] = "fail" if vulkan_failed else "pass"
        if name in skip:
            outcome = "SKIP"
        elif name in xfail:
            outcome = "XFAIL" if hrx_failed else "XPASS"
        else:
            outcome = "FAIL" if hrx_failed else "PASS"
        row["outcome"] = outcome
        log(f"{outcome}: HRX aggregate perplexity for {name}")
        unexpected = unexpected or outcome in ("FAIL", "XPASS")
    return unexpected


def run(args: argparse.Namespace) -> int:
    hrx_xfail_models = set(args.hrx_xfail_models)
    llama_perplexity = args.llama_perplexity.resolve()
    models_dir = args.models_dir.resolve()
    corpus_file = args.corpus_file.resolve()
    if not corpus_file.is_file():
        raise PerplexityBenchmarkError(f"Corpus file is missing: {corpus_file}")
    corpus_sha256 = sha256_file(corpus_file)
    models = resolve_models(args.model_manifest, models_dir, args.models)
    phases = (
        PerplexityPhase(
            name="HRX prefill-like",
            backend="hrx",
            device=args.hrx_device,
            regime_id="prefill-like",
            log=args.hrx_log,
        ),
        PerplexityPhase(
            name="Vulkan prefill-like",
            backend="vulkan",
            device=args.vulkan_device,
            regime_id="prefill-like",
            log=args.vulkan_log,
        ),
        PerplexityPhase(
            name="HRX decode-like",
            backend="hrx",
            device=args.hrx_device,
            regime_id="decode-like",
            log=args.hrx_log,
        ),
        PerplexityPhase(
            name="Vulkan decode-like",
            backend="vulkan",
            device=args.vulkan_device,
            regime_id="decode-like",
            log=args.vulkan_log,
        ),
    )
    batch_data = {
        "schema_version": 2,
        "devices": {"hrx": args.hrx_device, "vulkan": args.vulkan_device},
        "llama_perplexity": os.fspath(llama_perplexity),
        "regimes": REGIMES,
        "settings": {
            "extra_args": list(args.perplexity_arg),
            "max_perplexity_ratio": args.max_perplexity_ratio,
        },
        "corpus": {
            "name": corpus_file.name,
            "sha256": corpus_sha256,
            "bytes": corpus_file.stat().st_size,
        },
        "models": [
            {
                "model": model.spec.name,
                "file": model.spec.filename,
                "batch": args.batch_number,
                "regimes": {regime_id: {} for regime_id in REGIMES},
            }
            for model in models
        ],
    }

    # Precreate debug files so early failures still leave uploadable artifacts.
    for backend_log in (args.hrx_log, args.vulkan_log):
        backend_log.parent.mkdir(parents=True, exist_ok=True)
        if args.batched:
            backend_log.touch(exist_ok=True)
        else:
            backend_log.write_text("", encoding="utf-8")

    log_llama_perplexity_devices(llama_perplexity)

    with tempfile.TemporaryDirectory(
        dir=models_dir.parent,
        prefix="perplexity-state-",
    ) as state_root_name:
        state_root = Path(state_root_name)
        for phase in phases:
            log(f"Starting {phase.name} perplexity phase on {phase.device}")
            active_log = phase.log
            if args.batched:
                # Each phase appends only its own segment to the backend log.
                active_log = (
                    state_root / phase.regime_id / phase.backend / "perplexity.log"
                )
            try:
                measurements = run_phase(
                    phase,
                    active_log,
                    models,
                    llama_perplexity,
                    corpus_file,
                    args,
                )
            finally:
                if args.batched:
                    assert args.batch_number is not None
                    try:
                        append_batch_log(phase.log, active_log, args.batch_number)
                    except OSError as exc:
                        log(
                            "Warning: could not append "
                            f"{phase.backend} batch {args.batch_number} "
                            f"log to {phase.log}: {exc}"
                        )

            for row in batch_data["models"]:
                row["regimes"][phase.regime_id][phase.backend] = measurements[
                    row["model"]
                ]

        has_unexpected_outcomes = evaluate_models(
            batch_data, hrx_xfail_models,
            set(args.hrx_skip_models),
        )
        if args.batched:
            merged_count = merge_benchmark_output(args.output, batch_data)
            log(f"Merged {merged_count} model(s) from "
                f"batch {args.batch_number} into {args.output}")
        else:
            atomic_write_json(args.output, batch_data)
            log(f"Wrote {args.output}")

    if has_unexpected_outcomes:
        log(
            "Perplexity benchmark completed with unexpected FAIL or XPASS "
            "outcomes"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hrx-log", type=Path, required=True)
    parser.add_argument("--vulkan-log", type=Path, required=True)
    parser.add_argument("--hrx-device", default="HRX0")
    parser.add_argument("--vulkan-device", default="Vulkan0")
    parser.add_argument("--max-perplexity-ratio", type=float, default=1.10)
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--perplexity-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to llama-perplexity (repeatable).",
    )
    parser.add_argument("--hrx-xfail-models", nargs="*", required=True)
    parser.add_argument("--hrx-skip-models", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    ratio_is_finite = math.isfinite(args.max_perplexity_ratio)
    ratio_is_positive = args.max_perplexity_ratio > 0
    ratio_is_valid = ratio_is_finite and ratio_is_positive
    if not ratio_is_valid:
        parser.error("--max-perplexity-ratio must be finite and positive")
    if args.batched and args.batch_number is None:
        parser.error("--batch-number is required with --batched")
    if args.batch_number is not None and not args.batched:
        parser.error("--batch-number requires --batched")
    if args.batch_number is not None and args.batch_number < 1:
        parser.error("--batch-number must be a positive integer")
    try:
        return run(args)
    except (
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Perplexity benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
