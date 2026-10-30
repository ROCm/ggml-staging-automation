#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare pinned HRX llama builds on one GPU using a metadata-matched Qwen model.

This is an opt-in experiment. It leaves the release benchmark manifest, coverage,
and pass thresholds alone. The paired order balances drift while llama-bench's
own warmup precedes each three-sample measurement. A separate trace probe checks
that the Qwen metadata initializer actually ran on the target architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

from run_batched_benchmark import download_model, load_manifest


MODEL_ID = "qwen3-4b-instruct-2507"
ORDER = ("baseline", "candidate", "candidate", "baseline") * 2
FREE_MEMORY_PATTERN = re.compile(r"HRX0: .*\((\d+) MiB, (\d+) MiB free\)")
METADATA_KERNEL = "qwen3_moe:qwen_attention_metadata"


def run_bench(binary: Path, model: Path, output_dir: Path, label: str,
              *, trace: bool = False) -> list[dict]:
    prompt_tokens = "32" if trace else "128"
    repetitions = "1" if trace else "3"
    command = [
        str(binary), "-m", str(model), "-p", prompt_tokens, "-n", "32",
        "-b", "32", "-ub", "32", "-ngl", "99", "-dev", "HRX0",
        "-r", repetitions, "-o", "json",
    ]
    env = os.environ.copy()
    if trace:
        env["GGML_HRX_TRACE_DISPATCH_DIR"] = str(output_dir / "trace")
        env["GGML_HRX_TRACE_START_COMMAND"] = "1"
        env["GGML_HRX_TRACE_END_COMMAND"] = "1"
    print(f"Running {label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (output_dir / f"{label}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (output_dir / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; see saved logs")
    rows = json.loads(result.stdout)
    if len(rows) != 2:
        raise RuntimeError(f"{label} produced {len(rows)} cases; expected prompt and generation")
    return rows


def check_free_memory(binary: Path, model_size: int, output_dir: Path) -> str:
    result = subprocess.run([str(binary), "--list-devices"],
                            capture_output=True, text=True, check=True)
    (output_dir / "devices.txt").write_text(result.stdout + result.stderr,
                                            encoding="utf-8")
    match = FREE_MEMORY_PATTERN.search(result.stdout)
    if match is None:
        raise RuntimeError("Pinned HRX0 device or its free-memory reading is missing")
    free_mib = int(match.group(2))
    model_mib = (model_size + 1024 * 1024 - 1) // (1024 * 1024)
    if free_mib < model_mib + 1024:
        raise RuntimeError(f"HRX0 has {free_mib} MiB free; need at least {model_mib + 1024} MiB")
    print(result.stdout, flush=True)
    return match.group(0)


def case_name(row: dict) -> str:
    prompt = row["n_prompt"] > 0
    generation = row["n_gen"] > 0
    if prompt and not generation:
        return "prompt"
    if generation and not prompt:
        return "generation"
    raise RuntimeError(f"Unexpected llama-bench case: {row['n_prompt']}/{row['n_gen']}")


def summarize(runs: list[dict]) -> dict:
    summary = {}
    for case in ("prompt", "generation"):
        values = {}
        for variant in ("baseline", "candidate"):
            samples = [sample for run in runs if run["variant"] == variant
                       for row in run["rows"] if case_name(row) == case
                       for sample in row["samples_ts"]]
            values[variant] = {"median_tps": statistics.median(samples),
                               "samples_tps": samples}
        baseline = values["baseline"]["median_tps"]
        candidate = values["candidate"]["median_tps"]
        values["candidate_change_percent"] = 100 * (candidate / baseline - 1)
        summary[case] = values
    return summary


def check_trace(trace_path: Path) -> int:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    metadata = [row for row in rows if row["kernel"] == METADATA_KERNEL]
    output_names = {row["binding_name"] for row in metadata}
    expected_names = {"key_cache_indices", "value_cache_indices", "attention_mask"}
    if not expected_names.issubset(output_names):
        raise RuntimeError(f"Qwen metadata trace missing expected outputs: {output_names}")
    return len(metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binaries = {"baseline": args.baseline_bin.resolve(),
                "candidate": args.candidate_bin.resolve()}
    model = next(item for item in load_manifest(args.model_manifest).models if item.id == MODEL_ID)
    device = check_free_memory(binaries["baseline"], model.size_bytes, output_dir)
    model_path = download_model(model, args.model_dir.resolve())

    runs = []
    for index, variant in enumerate(ORDER, start=1):
        label = f"run-{index:02d}-{variant}"
        rows = run_bench(binaries[variant], model_path, output_dir, label)
        runs.append({"variant": variant, "rows": rows})

    run_bench(binaries["candidate"], model_path, output_dir, "metadata-probe", trace=True)
    trace_count = check_trace(output_dir / "trace" / "trace.jsonl")
    summary = summarize(runs)
    report = {"model_id": MODEL_ID, "model_sha256": model.sha256,
              "device": device, "settings": {"prompt": 128, "generation": 32,
              "batch": 32, "microbatch": 32, "repetitions_per_run": 3,
              "order": ORDER}, "metadata_trace_rows": trace_count,
              "summary": summary, "runs": runs}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    lines = ["# Qwen metadata copy comparison on gfx1151", "",
             f"Model: `{MODEL_ID}` (`{model.sha256}`).", "",
             f"Device: `{device}`. Metadata trace rows: {trace_count}.", "",
             "| Case | Baseline median tok/s | Candidate median tok/s | Change |",
             "| --- | ---: | ---: | ---: |"]
    for case, values in summary.items():
        lines.append(f"| {case} | {values['baseline']['median_tps']:.2f} | "
                     f"{values['candidate']['median_tps']:.2f} | "
                     f"{values['candidate_change_percent']:+.2f}% |")
    lines.extend(["", "Medians use 12 warmed samples per variant in ABBA × 2 order.", ""])
    markdown = "\n".join(lines)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
