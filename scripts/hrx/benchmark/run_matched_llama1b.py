#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare block metadata reuse with PR 123 on one GPU, preserving every measurement.

The baseline package is PR 123; this workflow builds the metadata reuse
candidate. This isolated follow-up removes differing runner hardware as a
timing confounder. It uses existing verified model/corpus download helpers and the
unchanged perplexity worker; llama-bench runs in alternating package order.

Three rounds retain all five repetitions for each shape/backend. JSON records
contain commands, source/package hashes, device state, and point-estimate
changes. Failed subprocesses cannot yield a success report. A temporary work
root holds large inputs and extracted binaries; uploaded evidence stays small.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from fetch_wikitext2 import fetch_corpus, sha256_file
from run_batched_benchmark import download_model, load_manifest

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "llama-3.2-1b-instruct"
SOURCES = {
    "before": {"run_id": 36658779016, "llama_cpp": "63805bc03cba4c276a0f67f6dd7677d2cdfe6f27"},
    "after": {"run_id": os.environ.get("GITHUB_RUN_ID"), "llama_cpp": "b0ff6238026a7c846a0d6ba9144e48c2d40ba38a"},
}
ROUNDS = 3
REPETITIONS = 5


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_optional(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def gpu_state() -> dict:
    paths = []
    for pattern in ("card*/device/gpu_busy_percent", "card*/device/mem_info_vram_used",
                    "card*/device/pp_dpm_sclk", "card*/device/pp_dpm_mclk",
                    "card*/device/hwmon/hwmon*/temp1_input", "card*/device/hwmon/hwmon*/power1_average"):
        paths.extend(Path("/sys/class/drm").glob(pattern))
    return {str(path): read_optional(path) for path in paths}


def select_gpu() -> tuple[str, dict]:
    """KFD GPU-node order defines ROCr ordinals; choose the required gfx1151."""
    nodes = sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"), key=lambda path: int(path.parent.name))
    topology = {str(path): read_optional(path) for path in nodes}
    gpu_ordinal = 0
    for properties in topology.values():
        fields = dict(line.split(maxsplit=1) for line in properties.splitlines() if " " in line)
        target = int(fields.get("gfx_target_version", "0"))
        if target == 110501:
            return str(gpu_ordinal), topology
        if target > 0:
            gpu_ordinal += 1
    raise RuntimeError(f"No gfx1151 GPU in KFD topology: {topology}")


def run_command(command: list[str], name: str, output: Path, env: dict, *, json_stdout: bool = False, timeout: int = 300, check: bool = True) -> dict:
    """Publish command provenance before execution and retain output on failure."""
    suffix = ".json" if json_stdout else ".stdout.log"
    record = {
        "command": command, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_state_before": gpu_state(), "ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"), "stdout": name + suffix, "stderr": name + ".stderr.log",
    }
    command_file = output / f"{name}.command.json"
    write_json(command_file, record)
    print(f"Running {name}: {command}", flush=True)
    started = time.monotonic()
    try:
        with (output / record["stdout"]).open("w") as stdout, (output / record["stderr"]).open("w") as stderr:
            with subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True) as process:
                try:
                    record["exit_code"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # The perplexity worker owns child processes; timeout ends the whole group.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["exit_code"] = None
        record["error"] = str(exc)
    record["duration_s"] = time.monotonic() - started
    record["gpu_state_after"] = gpu_state()
    write_json(command_file, record)
    failed = record["exit_code"] != 0
    checked_failure = check and failed
    if checked_failure:
        raise RuntimeError(f"{name} failed; see {command_file}")
    return record


def benchmark_rows(path: Path, side: str, device: str, expected_shapes: set[tuple[int, int]]) -> list[dict]:
    """Check external benchmark output before admitting it to timing summaries."""
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise RuntimeError(f"{path}: expected an array of benchmark results")
    shapes = {(row["n_depth"], row["n_gen"]) for row in rows}
    valid_shapes = shapes == expected_shapes and len(rows) == len(expected_shapes)
    if not valid_shapes:
        raise RuntimeError(f"{path}: incomplete or unexpected shapes: {shapes}")
    for row in rows:
        expected_commit = SOURCES[side]["llama_cpp"]
        commit = row.get("build_commit", "")
        commit_matches = bool(commit) and expected_commit.startswith(commit)
        rate = row.get("avg_ts")
        rate_valid = type(rate) in (int, float) and math.isfinite(rate) and rate > 0
        samples = row.get("samples_ts", [])
        samples_valid = len(samples) == REPETITIONS and all(math.isfinite(sample) and sample > 0 for sample in samples)
        identity_matches = row.get("devices") == device and row.get("n_prompt") == 0
        valid = commit_matches and rate_valid and samples_valid and identity_matches
        if not valid:
            raise RuntimeError(f"{path}: invalid measurement or mismatched source/device: {row}")
    return rows


def compare_throughput(measurements: list[dict]) -> list[dict]:
    comparisons = []
    for device in ("HRX0", "Vulkan0"):
        for depth, tokens in ((0, 128), (0, 512), (66, 256)):
            selected = [row for row in measurements if (row["device"], row["n_depth"], row["n_gen"]) == (device, depth, tokens)]
            rates = {side: [row["avg_ts"] for row in selected if row["side"] == side] for side in SOURCES}
            means = {side: statistics.mean(values) for side, values in rates.items()}
            paired = [100 * (after / before - 1) for before, after in zip(rates["before"], rates["after"])]
            comparisons.append({"device": device, "depth": depth, "tokens": tokens, "mean_tps": means,
                                "round_tps": rates, "paired_delta_percent": paired,
                                "delta_percent": 100 * (means["after"] / means["before"] - 1)})
    return comparisons


def compare_perplexity(output: Path) -> list[dict]:
    results = {side: json.loads((output / f"perplexity-{side}.json").read_text())["models"][0] for side in SOURCES}
    comparisons = []
    for regime in ("prefill-like", "decode-like"):
        for backend in ("hrx", "vulkan"):
            measurements = {side: row["regimes"][regime][backend] for side, row in results.items()}
            valid = all(row["status"] == "ok" for row in measurements.values())
            values = {side: row["ppl"]["value"] if row["status"] == "ok" else None for side, row in measurements.items()}
            delta = values["after"] - values["before"] if valid else None
            comparisons.append({"regime": regime, "backend": backend, "values": values,
                                "delta_absolute": delta, "delta_percent": 100 * delta / values["before"] if valid else None})
    return comparisons


def report(summary: dict) -> str:
    lines = [f"Comparison status: **{summary['status']}**", "", f"Runner: `{summary.get('runner_name')}`; host: `{summary.get('hostname')}`.",
             "Three interleaved rounds per package, five measurements per round; mean tok/s below averages the three rounds.",
             "Depth 66 prefills the KV cache before the measured 256-token decode; llama-bench excludes sampling/server overhead.", "",
             "| Backend | Depth / decode tokens | Before tok/s | After tok/s | Delta |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary.get("throughput", []):
        lines.append(f"| {row['device']} | {row['depth']} / {row['tokens']} | {row['mean_tps']['before']:.3f} | {row['mean_tps']['after']:.3f} | {row['delta_percent']:+.3f}% |")
    lines.extend(["", "| Backend | PPL regime | Before | After | Absolute delta | Percentage delta |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in summary.get("perplexity", []):
        if row["delta_absolute"] is None:
            lines.append(f"| {row['backend']} | {row['regime']} | unavailable | unavailable | unavailable | unavailable |")
        else:
            lines.append(f"| {row['backend']} | {row['regime']} | {row['values']['before']:.4f} | {row['values']['after']:.4f} | {row['delta_absolute']:+.4f} | {row['delta_percent']:+.4f}% |")
    if summary.get("error"):
        lines.extend(["", f"Error: {summary['error']}"])
    return "\n".join(lines) + "\n"


def experiment(args: argparse.Namespace, output: Path, summary: dict) -> None:
    pin, topology = select_gpu()
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=pin)
    manifest_path = ROOT / "benchmarks/hrx/model_manifest.json"
    model = next(model for model in load_manifest(manifest_path).models if model.id == MODEL_ID)
    summary.update({"gpu_pin": pin, "topology": topology, "model": dataclasses.asdict(model), "sources": SOURCES,
                    "runner_name": os.environ.get("RUNNER_NAME"), "hostname": platform.node(), "platform": platform.platform(),
                    "cpuinfo": read_optional(Path("/proc/cpuinfo")), "os_release": read_optional(Path("/etc/os-release")),
                    "automation_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
                    "rounds": ROUNDS, "repetitions": REPETITIONS, "measurements": []})
    run_command(["dpkg-query", "-W", "mesa-vulkan-drivers", "libdrm-amdgpu1"], "driver-packages", output, env, check=False)
    with tempfile.TemporaryDirectory(prefix="matched-llama1b-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        work = Path(temporary)
        model_path = download_model(model, work / "models")
        corpus = fetch_corpus(work / "corpus")
        summary["corpus_sha256"] = sha256_file(corpus)
        installs = {}
        environments = {}
        summary["packages"] = {}
        for side in SOURCES:
            archive = args.packages_dir.resolve() / side / "llama-cpp-linux-x86_64.tar.gz"
            install = work / side
            run_command([sys.executable, str(ROOT / "scripts/hrx/package/extract_release_package.py"),
                         "--archive", str(archive), "--package-root-name", "llama.cpp-install", "--output-dir", str(install)],
                        f"extract-{side}", output, env)
            summary["packages"][side] = {"archive_sha256": sha256_file(archive),
                "binary_sha256": {name: sha256_file(install / "bin" / name) for name in ("llama-bench", "llama-perplexity")},
                "library_sha256": {path.name: sha256_file(path) for path in sorted((install / "lib").glob("*.so*")) if path.is_file()}}
            installs[side] = install
            environments[side] = dict(env, LD_LIBRARY_PATH=str(install / "lib"))
            run_command([str(install / "bin/llama-bench"), "--list-devices"], f"devices-{side}", output, environments[side])
        # AB/BA/AB balances process order; each new process performs its default warmup.
        for round_index in range(ROUNDS):
            order = ("before", "after") if round_index % 2 == 0 else ("after", "before")
            for device in ("HRX0", "Vulkan0"):
                for side in order:
                    for shape_name, generation, depth in (("tg", "128,512", "0"), ("depth66-tg256", "256", "66")):
                        name = f"round{round_index}-{side}-{device}-{shape_name}"
                        command = [str(installs[side] / "bin/llama-bench"), "-m", str(model_path), "-p", "0",
                                   "-n", generation, "-d", depth, "-r", str(REPETITIONS), "-t", "4", "-ngl", "99", "-dev", device, "-o", "json"]
                        run_command(command, name, output, environments[side], json_stdout=True)
                        shapes = {(int(depth), int(tokens)) for tokens in generation.split(",")}
                        rows = benchmark_rows(output / f"{name}.json", side, device, shapes)
                        summary["measurements"].extend(dict(row, side=side, device=device, round=round_index) for row in rows)
                        write_json(output / "progress.json", summary)
        summary["throughput"] = compare_throughput(summary["measurements"])
        summary["perplexity_exit_codes"] = {}
        for side in SOURCES:
            command = [sys.executable, str(ROOT / "scripts/hrx/benchmark/run_perplexity_benchmark.py"),
                       "--llama-perplexity", str(installs[side] / "bin/llama-perplexity"),
                       "--model-manifest", str(manifest_path), "--models-dir", str(work / "models"),
                       "--corpus-file", str(corpus), "--output", str(output / f"perplexity-{side}.json"),
                       "--hrx-log", str(output / f"perplexity-{side}-hrx.log"),
                       "--vulkan-log", str(output / f"perplexity-{side}-vulkan.log"),
                       "--run-timeout-seconds", "300", "--hrx-xfail-models", "--models", model.name]
            result = run_command(command, f"perplexity-{side}", output, environments[side], timeout=1500, check=False)
            summary["perplexity_exit_codes"][side] = result["exit_code"]
        summary["perplexity"] = compare_perplexity(output)
        if any(code != 0 for code in summary["perplexity_exit_codes"].values()):
            raise RuntimeError("At least one unchanged perplexity check failed; raw outputs are retained")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "incomplete"}
    try:
        experiment(args, output, summary)
        summary["status"] = "complete"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(summary["error"], file=sys.stderr, flush=True)
    write_json(output / "summary.json", summary)
    rendered = report(summary)
    (output / "report.md").write_text(rendered)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a") as handle:
            handle.write(rendered)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
