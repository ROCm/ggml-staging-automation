#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure paired HRX/Vulkan prefill on the five smoke models affected by the experiment.

The batch driver owns model downloads. Each model runs three alternating backend
pairs on this runner, with five warmup-enabled samples per invocation. Raw JSON
and stderr remain in the debug artifact; the summary uses paired log ratios.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from benchmark_output import merge_benchmark_output
from run_perplexity_benchmark import resolve_models


MODEL_IDS = {
    "llama-3.2-1b-instruct", "llama-3.2-3b-instruct", "smollm3-3b-128k",
    "qwen3.5-0.8b", "qwen3.5-2b",
}


def parse_rate(text: str) -> float:
    rows = json.loads(text)
    if len(rows) != 1:
        raise ValueError("Expected exactly one llama-bench scenario")
    row = rows[0]
    prompt_ok = row["n_prompt"] == 512
    generation_ok = row["n_gen"] == 0
    shape_ok = prompt_ok and generation_ok
    if not shape_ok:
        raise ValueError("Expected pp512")
    rate = float(row["avg_ts"])
    finite = math.isfinite(rate)
    positive = rate > 0
    valid = finite and positive
    if not valid:
        raise ValueError("Expected finite positive throughput")
    return rate


def summarize(pairs: list[dict[str, float]]) -> dict:
    logs = [math.log(p["hrx"] / p["vulkan"]) for p in pairs]
    mean = statistics.mean(logs)
    margin = 4.30265273 * statistics.stdev(logs) / math.sqrt(3)
    return {
        "hrx_tps": statistics.mean(p["hrx"] for p in pairs),
        "vulkan_tps": statistics.mean(p["vulkan"] for p in pairs),
        "hrx_over_vulkan": math.exp(mean),
        "ratio_ci95_low": math.exp(mean - margin),
        "ratio_ci95_high": math.exp(mean + margin),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int, required=True)
    parser.add_argument("--hrx-xfail-models", nargs="*", required=True)
    parser.add_argument("--hrx-skip-models", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    unsupported_expectations = bool(args.hrx_xfail_models or args.hrx_skip_models)
    if unsupported_expectations:
        raise ValueError("The focused prefill experiment requires both backends to pass")
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    devices = subprocess.run([str(args.llama_bench), "--list-devices"],
                             text=True, capture_output=True, check=True)
    (args.logs_dir / f"devices-{args.batch_number}.txt").write_text(devices.stdout + devices.stderr)
    document = {"runner": os.environ.get("RUNNER_NAME", "local"), "prompt": 512,
                "repetitions": 5, "pair_order": ["HRX/Vulkan", "Vulkan/HRX", "HRX/Vulkan"],
                "models": []}
    for model in resolve_models(args.model_manifest, args.models_dir, args.models):
        if model.spec.id not in MODEL_IDS:
            continue
        pairs = []
        for pair, order in enumerate([("hrx", "vulkan"), ("vulkan", "hrx"), ("hrx", "vulkan")]):
            values = {}
            for backend in order:
                device = "HRX0" if backend == "hrx" else "Vulkan0"
                command = [str(args.llama_bench), "-m", str(model.path), "-dev", device,
                           "-ngl", "999", "-t", "16", "-b", "512", "-ub", "512",
                           "-p", "512", "-n", "0", "-r", "5", "-o", "json"]
                print(f"Prefill {model.spec.id} pair {pair}: {backend}", flush=True)
                result = subprocess.run(command, text=True, capture_output=True, timeout=600)
                stem = args.logs_dir / f"{model.spec.id}-{pair}-{backend}"
                Path(str(stem) + ".json").write_text(result.stdout)
                Path(str(stem) + ".stderr").write_text(result.stderr)
                Path(str(stem) + ".process.json").write_text(json.dumps({"argv": command, "returncode": result.returncode}))
                result.check_returncode()
                values[backend] = parse_rate(result.stdout)
            pairs.append(values)
        document["models"].append({"model": model.spec.id, **summarize(pairs)})
    merge_benchmark_output(args.output, document)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write("\n### Paired pp512 prefill: HRX / Vulkan on this runner\n\n")
            summary.write("Three alternating pairs, five repetitions each. Per-comparison 95% t intervals; not a baseline-versus-patch speedup.\n\n")
            summary.write("| Model | HRX tok/s | Vulkan tok/s | Ratio | 95% interval |\n| --- | ---: | ---: | ---: | ---: |\n")
            for row in document["models"]:
                summary.write(f"| {row['model']} | {row['hrx_tps']:.1f} | {row['vulkan_tps']:.1f} | {row['hrx_over_vulkan']:.3f}x | {row['ratio_ci95_low']:.3f}-{row['ratio_ci95_high']:.3f}x |\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
