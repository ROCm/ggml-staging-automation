#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare PR 108's original and repaired packages on one gfx1151 runner.

Use the normal HRX/Vulkan perplexity worker and its enforced numerical verdicts
for the single pinned Llama model. Original-package failures are evidence;
every repaired-package repetition must pass. Inputs are verified by the existing
manifest downloader and corpus loader, then removed from the self-hosted runner.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from fetch_wikitext2 import fetch_corpus
from run_batched_benchmark import download_model, load_manifest


MODEL = "extra.Llama-3.2-1B-Instruct-GGUF"
REPO_ROOT = Path(__file__).resolve().parents[3]


def select_gpu() -> dict:
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    gpu_ordinal = 0
    for node in sorted(nodes.iterdir(), key=lambda path: int(path.name)):
        properties = dict(line.split() for line in (node / "properties").read_text().splitlines())
        target = int(properties["gfx_target_version"])
        if target == 0:
            continue
        if target == 110501:
            os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_ordinal)
            return {"target": "gfx1151", "rocr_ordinal": gpu_ordinal, "properties": properties}
        gpu_ordinal += 1
    raise RuntimeError("No gfx1151 GPU in the runner's KFD topology")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-package", type=Path, required=True)
    parser.add_argument("--fixed-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "runner": os.environ.get("RUNNER_NAME", "local"),
        "hostname": platform.node(),
        "gpu": select_gpu(),
        "model": MODEL,
        "original_run": "39413873090",
        "repetitions": args.repetitions,
        "commits": {
            name: subprocess.check_output(
                ["git", "rev-parse", ref], cwd=REPO_ROOT, text=True,
            ).strip()
            for name, ref in (("staging", "HEAD"), ("llama.cpp", "HEAD:llama.cpp"), ("hrx-system", "HEAD:hrx-system"))
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = REPO_ROOT / "benchmarks/hrx/model_manifest.json"
    model = next(model for model in load_manifest(manifest).models if model.name == MODEL)
    results = []
    fixed_failed = False
    with tempfile.TemporaryDirectory(prefix="pr108-perplexity-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        inputs = Path(temporary)
        models = inputs / "models"
        download_model(model, models)
        corpus = fetch_corpus(inputs / "corpus")
        for repetition in range(1, args.repetitions + 1):
            for variant, package in (("original", args.original_package), ("fixed", args.fixed_package)):
                prefix = output / f"{variant}-{repetition}"
                command = [
                    sys.executable, str(Path(__file__).with_name("run_perplexity_benchmark.py")),
                    "--llama-perplexity", str(package.resolve() / "bin/llama-perplexity"),
                    "--model-manifest", str(manifest), "--models-dir", str(models),
                    "--corpus-file", str(corpus), "--output", str(prefix.with_suffix(".json")),
                    "--hrx-log", str(prefix.with_suffix(".hrx.log")),
                    "--vulkan-log", str(prefix.with_suffix(".vulkan.log")),
                    "--hrx-xfail-models", "--models", MODEL,
                ]
                with prefix.with_suffix(".console.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                results.append({"variant": variant, "repetition": repetition, "exit_code": result.returncode})
                repaired_package = variant == "fixed"
                worker_failed = result.returncode != 0
                fixed_failed = fixed_failed or (repaired_package and worker_failed)
                print(f"{variant} repetition {repetition}: exit {result.returncode}", flush=True)
                (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        f"Runner: `{metadata['runner']}`; host: `{metadata['hostname']}`; GPU: gfx1151",
        f"\nModel: `{MODEL}`\n",
        "| Package | Repetition | Perplexity verdict |",
        "| --- | --- | --- |",
    ]
    for result in results:
        verdict = "PASS" if result["exit_code"] == 0 else "FAIL"
        lines.append(f"| {result['variant']} | {result['repetition']} | {verdict} |")
    summary = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a") as handle:
            handle.write(summary)
    return int(fixed_failed)


if __name__ == "__main__":
    raise SystemExit(main())
