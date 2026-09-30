#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure a standalone kernel change against its parent on one smoke GPU.

Before and after are separately packaged builds, run AB/BA/AB for each model.
The standard Lemonade worker owns both throughput scenarios and their warmups;
the standard perplexity worker owns the unchanged corpus regimes and checks.
Only scheduling and artifact paths differ from the release smoke workflow.

One verified model stays resident on disk until both packages finish, bounding
disk usage and keeping paired runs close in time. Every command and raw result
survives in the output directory, including failures; matched-smoke.json is the
input to the independent report writer. Worker subprocesses install cooperative
termination so Lemonade's separately sessioned daemon cannot survive a timeout.
An unsafe cleanup aborts the experiment rather than contaminating later runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time

from fetch_wikitext2 import fetch_corpus, sha256_file
from run_batched_benchmark import ModelSpec, download_model, load_manifest, select_models
from run_matched_llama1b import gpu_state, read_optional, select_gpu

ROOT = Path(__file__).resolve().parents[3]
SIDES = ("before", "after")
BACKENDS = ("hrx", "vulkan")
SCENARIOS = ("chat-short", "chat-long-output")
ORDERS = (("before", "after"), ("after", "before"), ("before", "after"))


class UnsafeCleanup(RuntimeError):
    """A live process or daemon prevents further uncontaminated measurement."""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def port_in_use() -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", 13305)) == 0


def group_alive(group: int) -> bool:
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            is_group = int(fields[2]) == group
            is_live = fields[0] != "Z"
            belongs_to_live_group = is_group and is_live
            if belongs_to_live_group:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def clean_group(group: int) -> None:
    for sig, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if not group_alive(group):
            return
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not group_alive(group):
                return
            time.sleep(0.1)
    raise UnsafeCleanup(f"Process group {group} survived termination")


def worker_main(kind: str, arguments: list[str]) -> int:
    """Retain standard worker behavior while giving its daemon a cleanup owner."""
    def interrupted(signum: int, _frame: object) -> None:
        # A single interruption unwinds the worker's existing daemon finally block.
        signal.signal(signum, signal.SIG_IGN)
        raise RuntimeError(f"Worker interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    module = importlib.import_module(f"run_{kind}_benchmark")
    if kind == "lemonade":
        original_start = module.start_lemond
        daemon_groups: list[int] = []

        def tracked_start(*args: object, **kwargs: object) -> subprocess.Popen:
            process = original_start(*args, **kwargs)
            daemon_groups.append(process.pid)
            write_json(Path(os.environ["MATCHED_DAEMON_GROUPS"]), daemon_groups)
            return process

        module.start_lemond = tracked_start
    sys.argv = [module.__file__, *arguments]
    return module.main()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_command(command: list[str], directory: Path, env: dict[str, str], *,
                timeout: int, worker: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"),
              "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"), "gpu_state_before": gpu_state()}
    command_path = directory / "command.json"
    write_json(command_path, record)
    daemon_file = directory / "daemon-groups.json"
    child_env = dict(env, MATCHED_DAEMON_GROUPS=str(daemon_file))
    started = time.monotonic()
    process = None
    print("Running " + " ".join(command), flush=True)
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, env=child_env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                record["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["error"] = f"Timed out after {timeout} seconds"
                stop_process(process)
                record["exit_code"] = process.returncode
    except OSError as exc:
        record.update(exit_code=None, error=str(exc))
        if process is not None:
            stop_process(process)
    except BaseException:
        if process is not None:
            stop_process(process)
        raise
    finally:
        try:
            groups = json.loads(daemon_file.read_text()) if daemon_file.exists() else []
            if process is not None:
                groups.append(process.pid)
            leaked_groups = [group for group in groups if group_alive(group)]
            for group in leaked_groups:
                clean_group(group)
            if leaked_groups:
                raise UnsafeCleanup(f"Worker leaked live process groups: {leaked_groups}; stopped experiment")
            occupied_worker_port = worker and port_in_use()
            if occupied_worker_port:
                raise UnsafeCleanup("Lemonade port 13305 remains occupied after worker exit")
        except Exception as exc:
            record["cleanup_error"] = str(exc)
            raise
        finally:
            record["duration_s"] = time.monotonic() - started
            record["gpu_state_after"] = gpu_state()
            write_json(command_path, record)
    return record


def positive(value: object) -> bool:
    is_number = type(value) in (int, float)
    return is_number and math.isfinite(value) and value > 0


def read_throughput(directory: Path, model: ModelSpec, side: str, round_index: int,
                    output: Path) -> list[dict]:
    """Validate external worker JSON and raw samples at their owning boundary."""
    records = []
    for backend in BACKENDS:
        path = directory / f"benchmark-{backend}.json"
        response_path = directory / f"responses-{backend}.jsonl"
        document = json.loads(path.read_text())
        rows = document["models"]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one model")
        if rows[0]["model"] != model.name:
            raise ValueError(f"{path}: expected only {model.name}")
        row = rows[0]
        configuration = row["config"]
        correct_runs = configuration["measurement_runs"] == 3
        correct_warmup = configuration["warmup_runs"] == 1
        expected_config = correct_runs and correct_warmup
        if not expected_config:
            raise ValueError(f"{path}: altered warmup/repetition count")
        results = row["results"]
        if len(results) != 1:
            raise ValueError(f"{path}: expected one backend result")
        if results[0]["backend"] != backend:
            raise ValueError(f"{path}: unexpected backend results")
        result = results[0]
        if result["backend_args"] != "--ignore-eos":
            raise ValueError(f"{path}: unexpected backend arguments")
        scenarios = result["scenarios"]
        correct_names = {item["name"] for item in scenarios} == set(SCENARIOS)
        correct_count = len(scenarios) == len(SCENARIOS)
        correct_scenarios = correct_names and correct_count
        if not correct_scenarios:
            raise ValueError(f"{path}: incomplete or duplicate scenarios")
        responses = [json.loads(line) for line in response_path.read_text().splitlines() if line.strip()]
        if len(responses) != 6:
            raise ValueError(f"{response_path}: expected six measured responses")
        for scenario in scenarios:
            name = scenario["name"]
            samples = [item for item in responses if item["scenario"] == name]
            identities_match = all((item["model"], item["backend"]) == (model.name, backend) for item in samples)
            sample_numbers = sorted(item["run_number"] for item in samples)
            correct_sample_count = len(samples) == 3
            correct_sample_numbers = sample_numbers == [1, 2, 3]
            correct_samples = correct_sample_count and correct_sample_numbers and identities_match
            rates = [item["tps"] for item in samples]
            valid_rates = all(positive(rate) for rate in rates)
            no_failed_runs = scenario["failed_runs"] == 0
            no_total_failure = not scenario.get("all_runs_failed", False)
            no_failures = no_failed_runs and no_total_failure
            valid = correct_samples and valid_rates and no_failures
            if not valid:
                raise ValueError(f"{path}: invalid {name} samples or outcomes")
            mean = scenario["tps"]["mean"]
            mean_positive = positive(mean)
            mean_matches_samples = math.isclose(mean, statistics.fmean(rates), rel_tol=1e-9) if mean_positive else False
            valid_mean = mean_positive and mean_matches_samples
            if not valid_mean:
                raise ValueError(f"{path}: {name} mean does not match raw responses")
            records.append({"model": model.id, "scenario": name, "variant": side, "backend": backend,
                            "round": round_index, "status": "ok", "samples_tps": rates, "mean_tps": mean,
                            "raw_json": str(path.relative_to(output)),
                            "raw_responses": str(response_path.relative_to(output))})
    return records


def read_perplexity(directory: Path, model: ModelSpec, side: str, output: Path) -> list[dict]:
    path = directory / "perplexity.json"
    document = json.loads(path.read_text())
    rows = document["models"]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one model")
    if rows[0]["model"] != model.name:
        raise ValueError(f"{path}: expected only {model.name}")
    rows = rows[0]["regimes"]
    if set(rows) != {"prefill-like", "decode-like"}:
        raise ValueError(f"{path}: unexpected perplexity regimes")
    records = []
    for regime, measurements in rows.items():
        for backend in BACKENDS:
            measurement = measurements[backend]
            ok = measurement["status"] == "ok"
            value = measurement["ppl"]["value"] if ok else None
            valid = ok and positive(value)
            if not valid:
                raise ValueError(f"{path}: invalid {regime}/{backend} perplexity")
            records.append({"model": model.id, "variant": side, "regime": regime, "backend": backend,
                            "status": "ok", "value": value,
                            "uncertainty": measurement["ppl"]["uncertainty"],
                            "raw_json": str(path.relative_to(output))})
    return records


def worker_command(kind: str, options: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", kind, *options]


def full_commit(value: str) -> str:
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("Expected a full lowercase Git commit SHA")
    return value


def device_identity(text: str) -> dict[str, str]:
    """Require both packages' external enumeration to name one matching GPU."""
    rows = re.findall(r"^\s*((?:HRX|Vulkan)\d+):\s*(.+)$", text, flags=re.MULTILINE)
    correct_count = len(rows) == 2
    correct_devices = {row[0] for row in rows} == {"HRX0", "Vulkan0"}
    valid = correct_count and correct_devices
    if not valid:
        raise ValueError(f"Expected exactly HRX0 and Vulkan0, got {rows}")
    names = []
    identities = {}
    for device, description in rows:
        if "gfx1151" not in description.lower():
            raise ValueError(f"{device} does not identify a gfx1151 GPU")
        name = description.split("(", 1)[0].strip().lower().removeprefix("amd ")
        names.append(name)
        identities[device] = re.sub(r"\d+ MiB free", "<free> MiB free", description)
    if names[0] != names[1]:
        raise ValueError(f"HRX and Vulkan identify different GPU hardware: {rows}")
    return identities


def checkpoint(output: Path, data: dict) -> None:
    write_json(output / "progress.json", data)
    write_json(output / "matched-smoke.json", data)


def experiment(args: argparse.Namespace, output: Path, data: dict) -> None:
    if port_in_use():
        raise UnsafeCleanup("Lemonade port 13305 is occupied before the experiment")
    manifest_path = ROOT / "benchmarks/hrx/model_manifest.json"
    models = select_models(load_manifest(manifest_path), "smoke")
    if len(models) != 25:
        raise RuntimeError(f"Expected 25 smoke models, found {len(models)}")
    if any(model.hrx_expected_results for model in models):
        raise RuntimeError("Matched smoke requires all models with ordinary PASS expectations")
    # These mutable checkouts are external inputs to the packaged experiment.
    actual_hrx = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:hrx-system"], text=True).strip()
    lemonade_source = args.lemonade_build_dir.resolve().parent
    actual_lemonade = subprocess.check_output(["git", "-C", str(lemonade_source), "rev-parse", "HEAD"], text=True).strip()
    hrx_matches = actual_hrx == args.hrx_system_commit
    lemonade_matches = actual_lemonade == args.lemonade_commit
    sources_match = hrx_matches and lemonade_matches
    if not sources_match:
        raise RuntimeError(f"Source checkout mismatch: HRX={actual_hrx}, Lemonade={actual_lemonade}")
    pin, topology = select_gpu()
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=pin)
    sources = {side: {"llama_cpp": getattr(args, f"{side}_commit"), "run_id": getattr(args, f"{side}_run_id")}
               for side in SIDES}
    sources.update(hrx_system=args.hrx_system_commit, lemonade=args.lemonade_commit)
    data.update(expected_models=[model.id for model in models])
    metadata = data["metadata"]
    metadata.update(models=[dataclasses.asdict(model) for model in models], sources=sources,
                    runner_name=os.environ.get("RUNNER_NAME"), hostname=platform.node(), platform=platform.platform(),
                    gpu_pin=pin, topology=topology, cpuinfo=read_optional(Path("/proc/cpuinfo")),
                    meminfo=read_optional(Path("/proc/meminfo")), os_release=read_optional(Path("/etc/os-release")),
                    automation_commit=os.environ.get("GITHUB_SHA"), run_id=os.environ.get("GITHUB_RUN_ID"),
                    run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"), manifest_sha256=sha256_file(manifest_path),
                    orders=ORDERS, packages={})
    checkpoint(output, data)
    run_command(["dpkg-query", "-W", "mesa-vulkan-drivers", "libdrm-amdgpu1"], output / "driver-packages", env, timeout=60)
    with tempfile.TemporaryDirectory(prefix="matched-smoke-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        work = Path(temporary)
        corpus = fetch_corpus(work / "corpus")
        metadata["corpus_sha256"] = sha256_file(corpus)
        installs, environments = {}, {}
        for side in SIDES:
            archive = args.packages_dir.resolve() / side / "llama-cpp-linux-x86_64.tar.gz"
            install = work / side
            command = [sys.executable, str(ROOT / "scripts/hrx/package/extract_release_package.py"),
                       "--archive", str(archive), "--package-root-name", "llama.cpp-install", "--output-dir", str(install)]
            result = run_command(command, output / f"extract-{side}", env, timeout=300)
            if result["exit_code"] != 0:
                raise RuntimeError(f"Could not extract {side} package")
            installs[side] = install
            environments[side] = dict(env, LD_LIBRARY_PATH=str(install / "lib"))
            metadata["packages"][side] = {"archive_sha256": sha256_file(archive),
                "binary_sha256": {name: sha256_file(install / "bin" / name)
                                  for name in ("llama-server", "llama-bench", "llama-perplexity")},
                "library_sha256": {str(path.relative_to(install)): sha256_file(path)
                                   for path in sorted((install / "lib").rglob("*")) if path.is_file()}}
            for check, option in (("version", "--version"), ("devices", "--list-devices")):
                directory = output / f"{check}-{side}"
                result = run_command([str(install / "bin/llama-server"), option], directory, environments[side], timeout=300)
                if result["exit_code"] != 0:
                    raise RuntimeError(f"{side} package {check} check failed")
                text = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
                is_version_check = check == "version"
                commit_matches = sources[side]["llama_cpp"][:7] in text
                wrong_commit = is_version_check and not commit_matches
                if wrong_commit:
                    raise RuntimeError(f"{side} package {check} does not match requested source/devices")
                if check == "devices":
                    identity = device_identity(text)
                    metadata["packages"][side]["devices"] = identity
                    if side == "after":
                        if identity != metadata["packages"]["before"]["devices"]:
                            raise RuntimeError("Packages enumerate different GPU hardware")
            checkpoint(output, data)
        for model in models:
            model_root = work / "models"
            try:
                download_model(model, model_root)
                for round_index, order in enumerate(ORDERS):
                    for side in order:
                        directory = output / "raw" / model.id / f"round{round_index}" / side
                        options = ["--lemonade-build-dir", str(args.lemonade_build_dir.resolve()),
                                   "--llama-server", str(installs[side] / "bin/llama-server"),
                                   "--models-dir", str(model_root)]
                        for backend in BACKENDS:
                            options.extend([f"--{backend}-output", str(directory / f"benchmark-{backend}.json"),
                                            f"--{backend}-server-log", str(directory / f"lemond-{backend}.log"),
                                            f"--{backend}-response-log", str(directory / f"responses-{backend}.jsonl")])
                        options.extend(["--hrx-xfail-models", "--models", model.name])
                        result = run_command(worker_command("lemonade", options), directory, environments[side], timeout=1800, worker=True)
                        data["commands"].append({"model": model.id, "variant": side, "round": round_index,
                                                "kind": "throughput", "path": str((directory / "command.json").relative_to(output)),
                                                "exit_code": result["exit_code"]})
                        try:
                            if result["exit_code"] != 0:
                                raise RuntimeError(f"Worker exited {result['exit_code']}")
                            data["records"].extend(read_throughput(directory, model, side, round_index, output))
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                            data["failures"].append({"model": model.id, "variant": side, "round": round_index,
                                                     "kind": "throughput", "error": str(exc)})
                        checkpoint(output, data)
                for side in SIDES:
                    directory = output / "raw" / model.id / "perplexity" / side
                    options = ["--llama-perplexity", str(installs[side] / "bin/llama-perplexity"),
                               "--model-manifest", str(manifest_path), "--models-dir", str(model_root),
                               "--corpus-file", str(corpus), "--output", str(directory / "perplexity.json"),
                               "--hrx-log", str(directory / "perplexity-hrx.log"),
                               "--vulkan-log", str(directory / "perplexity-vulkan.log"),
                               "--hrx-xfail-models", "--models", model.name]
                    result = run_command(worker_command("perplexity", options), directory, environments[side], timeout=7500, worker=True)
                    data["commands"].append({"model": model.id, "variant": side, "kind": "perplexity",
                                            "path": str((directory / "command.json").relative_to(output)), "exit_code": result["exit_code"]})
                    try:
                        data["perplexity"].extend(read_perplexity(directory, model, side, output))
                        if result["exit_code"] != 0:
                            raise RuntimeError(f"Worker exited {result['exit_code']}")
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        data["failures"].append({"model": model.id, "variant": side, "kind": "perplexity", "error": str(exc)})
                    checkpoint(output, data)
            finally:
                if model_root.exists():
                    shutil.rmtree(model_root)
        if data["failures"]:
            raise RuntimeError("Smoke measurements contain failures; see failures and raw artifacts")
        complete_throughput = len(data["records"]) == 600
        complete_perplexity = len(data["perplexity"]) == 200
        complete = complete_throughput and complete_perplexity
        if not complete:
            raise RuntimeError("Incomplete model/scenario coverage")


def main() -> int:
    if sys.argv[1:2] == ["--worker"]:
        return worker_main(sys.argv[2], sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages-dir", type=Path, required=True)
    parser.add_argument("--lemonade-build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for side in SIDES:
        parser.add_argument(f"--{side}-commit", type=full_commit, required=True)
        parser.add_argument(f"--{side}-run-id", type=int)
    parser.add_argument("--hrx-system-commit", type=full_commit, required=True)
    parser.add_argument("--lemonade-commit", type=full_commit, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        if any(output.iterdir()):
            parser.error("--output-dir must be empty to prevent mixing different experiment attempts")
    output.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "status": "incomplete", "expected_models": [], "scenarios": list(SCENARIOS),
            "rounds": 3, "records": [], "perplexity": [], "metadata": {}, "commands": [], "failures": []}
    checkpoint(output, data)

    def interrupted(signum: int, _frame: object) -> None:
        signal.signal(signum, signal.SIG_IGN)
        raise RuntimeError(f"Experiment interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        experiment(args, output, data)
        data["status"] = "complete"
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
        print(data["error"], file=sys.stderr, flush=True)
    checkpoint(output, data)
    return 0 if data["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
