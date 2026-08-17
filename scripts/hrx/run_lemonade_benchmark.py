#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run HRX and Vulkan release benchmarks through sequential Lemonade daemons."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_BACKEND_ARGS = "--ignore-eos"


@dataclass
class BenchmarkPhase:
    name: str
    backend: str
    device: str
    output: Path
    server_log: Path
    response_log: Path


def log(message: str) -> None:
    print(message, flush=True)


def remove_state_root(state_root: Path) -> None:
    if state_root.exists():
        shutil.rmtree(state_root)


def prepare_state(state_root: Path) -> tuple[Path, Path, Path]:
    """Keep Lemonade and model caches under one disposable root for cleanup."""
    remove_state_root(state_root)
    cache_dir = state_root / "lemonade"
    hf_home = state_root / "huggingface"
    runtime_dir = state_root / "runtime"
    for path in (cache_dir, hf_home, runtime_dir):
        path.mkdir(parents=True)
    runtime_dir.chmod(0o700)
    return cache_dir, hf_home, runtime_dir


def request(port: int, path: str, timeout: float = 10.0) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    return response.status, payload


def request_json(port: int, path: str) -> dict[str, Any]:
    status, payload = request(port, path)
    if status != 200:
        raise RuntimeError(f"GET {path} returned HTTP {status}")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object from GET {path}")
    return value


def wait_for_live(process: subprocess.Popen[str], port: int) -> None:
    """Avoid racing Lemonade CLI requests against daemon startup."""
    deadline = time.monotonic() + 60.0
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"lemond exited with status {return_code} before becoming ready"
            )
        try:
            status, _ = request(port, "/live", timeout=2.0)
            if status == 200:
                return
            last_error = f"GET /live returned HTTP {status}"
        except (
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"lemond did not become ready within 60 seconds: {last_error}")


def stop_server(process: subprocess.Popen[str] | None) -> int | None:
    """Stop the daemon and flush its log before verification and state removal."""
    if process is None:
        return None
    if process.poll() is not None:
        return process.returncode

    process.send_signal(signal.SIGTERM)
    try:
        return process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=5)


def start_lemond(
    lemonade_build_dir: Path,
    cache_dir: Path,
    log_path: Path,
    env: dict[str, str],
    port: int,
) -> subprocess.Popen[str]:
    """Start the daemon that owns Lemonade configuration and backend processes."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.fspath(lemonade_build_dir / "lemond"),
        os.fspath(cache_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    log("++ " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def summarize_benchmark(
    data: dict[str, Any],
    expected_backend: str,
    expected_models: list[str],
) -> tuple[int, int]:
    """Check batch identity and count scenarios that report failures."""
    scenario_count = 0
    failed_scenario_count = 0
    model_names: set[str] = set()
    try:
        for model in data["models"]:
            model_name = model["model"]
            if model_name in model_names:
                raise RuntimeError(
                    f"Lemonade benchmark has duplicate model {model_name!r}"
                )
            model_names.add(model_name)
            model_scenario_count = 0
            for result in model["results"]:
                backend = result["backend"]
                if backend != expected_backend:
                    raise RuntimeError(
                        f"Lemonade reported backend {backend!r}; expected "
                        f"{expected_backend!r}"
                    )
                for scenario in result["scenarios"]:
                    failed_runs = scenario["failed_runs"]
                    if type(failed_runs) is not int or failed_runs < 0:
                        raise RuntimeError(
                            "Lemonade scenario failed_runs must be a "
                            "non-negative integer"
                        )
                    all_runs_failed = scenario.get("all_runs_failed", False)
                    if type(all_runs_failed) is not bool:
                        raise RuntimeError(
                            "Lemonade scenario all_runs_failed must be a "
                            "Boolean when present"
                        )
                    if failed_runs or all_runs_failed:
                        failed_scenario_count += 1
                    model_scenario_count += 1
                    scenario_count += 1
            if not model_scenario_count:
                raise RuntimeError(
                    f"Lemonade model {model_name!r} has no scenarios"
                )
        expected_model_names = set(expected_models)
        if model_names != expected_model_names:
            missing_models = sorted(expected_model_names - model_names)
            unexpected_models = sorted(model_names - expected_model_names)
            raise RuntimeError(
                "Lemonade benchmark model coverage mismatch: "
                f"missing={missing_models!r}; unexpected={unexpected_models!r}"
            )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Lemonade benchmark report is malformed") from exc
    return scenario_count, failed_scenario_count


def run_benchmark(
    executable: Path,
    backend: str,
    output: Path,
    response_log: Path,
    models: list[str],
    *,
    auto_pull: bool,
    env: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Run and summarize one backend's complete benchmark suite."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [
        os.fspath(executable),
        "bench",
        "--backend",
        backend,
        "--warmup",
        "1",
        "--runs",
        "3",
        f"--llamacpp-args={BENCHMARK_BACKEND_ARGS}",
        "--response-log",
        os.fspath(response_log),
        "--output",
        os.fspath(output),
    ]
    if auto_pull:
        command.append("--auto-pull")
    command.extend(models)
    log("++ " + " ".join(command))
    subprocess.run(command, env=env, check=True)

    if not output.is_file():
        raise RuntimeError(f"Lemonade benchmark did not write {output}")
    data = json.loads(output.read_text(encoding="utf-8"))
    scenario_count, failed_scenario_count = summarize_benchmark(
        data,
        expected_backend=backend,
        expected_models=models,
    )
    log(
        f"Summarized {scenario_count} {backend} scenario(s); "
        f"{failed_scenario_count} reported failures in {output}"
    )
    return data, failed_scenario_count


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON document in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as temporary_file:
        json.dump(data, temporary_file, indent=2)
        temporary_file.write("\n")
    os.replace(temporary_path, path)


def merge_benchmark_output(
    cumulative_output: Path, batch_data: dict[str, Any]
) -> int:
    """Atomically append one trusted backend batch to cumulative output."""
    if cumulative_output.exists():
        merged_data = json.loads(cumulative_output.read_text(encoding="utf-8"))
        merged_data["models"].extend(batch_data["models"])
    else:
        merged_data = batch_data

    atomic_write_json(cumulative_output, merged_data)
    return len(batch_data["models"])


def append_batch_log(destination: Path, source: Path, batch_number: int) -> None:
    """Append one temporary log between stable batch-number delimiters."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = f"batch {batch_number}"
    with destination.open("a", encoding="utf-8") as destination_file:
        destination_file.write(f"===== BEGIN {label} =====\n")
        if source.is_file():
            with source.open(
                encoding="utf-8", errors="replace"
            ) as source_file:
                shutil.copyfileobj(source_file, destination_file)
        destination_file.write(f"\n===== END {label} =====\n")


def set_lemonade_config_values(
    executable: Path,
    llama_server: Path,
    device: str,
    *,
    models_dir: Path | None = None,
    env: dict[str, str],
    port: int,
) -> None:
    """Pin both backends and select the device for the current phase."""
    command = [
        os.fspath(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "config",
        "set",
        f"llamacpp.hrx_bin={llama_server}",
        f"llamacpp.vulkan_bin={llama_server}",
        f"llamacpp.device={device}",
        "no_fetch_executables=true",
        "log_level=debug",
    ]
    if models_dir is not None:
        command.append(f"extra_models_dir={models_dir}")
    log("++ " + " ".join(command))
    subprocess.run(command, env=env, check=True)
    config = request_json(port, "/internal/config")
    configured_models_dir = config.get("extra_models_dir")
    if (
        Path(config["llamacpp"]["hrx_bin"]).resolve() != llama_server
        or Path(config["llamacpp"]["vulkan_bin"]).resolve() != llama_server
        or config["llamacpp"]["device"] != device
        or config["no_fetch_executables"] is not True
        or config["log_level"] != "debug"
        or (
            models_dir is not None
            and (
                not isinstance(configured_models_dir, str)
                or Path(configured_models_dir).resolve() != models_dir
            )
        )
    ):
        raise RuntimeError("Lemonade did not retain the requested configuration")
    log(
        "Verified Lemonade configuration: "
        f"llamacpp.hrx_bin={llama_server}, "
        f"llamacpp.vulkan_bin={llama_server}, "
        f"llamacpp.device={device}, no_fetch_executables=true, log_level=debug"
        + (
            f", extra_models_dir={models_dir}"
            if models_dir is not None
            else ""
        )
    )


def verify_configured_executable(
    *,
    phase: BenchmarkPhase,
    phases: tuple[BenchmarkPhase, ...],
    cache_dir: Path,
    llama_server: Path,
) -> None:
    """Verify log and cache evidence for the pinned binary and selected device."""
    log_text = phase.server_log.read_text(encoding="utf-8", errors="replace")
    positive_text = (
        "Fetching executable artifacts is disabled; using installed "
        f"llamacpp:{phase.backend} backend at {llama_server}"
    )
    managed_backend_dir = cache_dir / "bin" / "llamacpp" / phase.backend
    used_configured_executable = positive_text in log_text
    used_selected_device = f"--device {phase.device}" in log_text
    used_other_device = any(
        f"--device {other_phase.device}" in log_text
        for other_phase in phases
        if other_phase.device != phase.device
    )
    created_managed_backend = managed_backend_dir.exists()
    installed_llama_server = "Installing llama-server" in log_text
    if (
        not used_configured_executable
        or not used_selected_device
        or used_other_device
        or created_managed_backend
        or installed_llama_server
    ):
        raise RuntimeError(
            f"Could not prove that Lemonade used only the configured {phase.backend} "
            f"executable on {phase.device} without installing another backend"
        )
    log(
        f"Verified {phase.backend} used {llama_server} on {phase.device} without "
        "downloading another backend executable."
    )


def log_llama_server_devices(llama_server: Path, env: dict[str, str]) -> None:
    """Record the devices llama-server enumerates before benchmarking."""
    result = subprocess.run(
        [os.fspath(llama_server), "--list-devices"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(f"llama-server --list-devices (exit {result.returncode}):")
    log(result.stdout.rstrip())


def run(args: argparse.Namespace) -> int:
    port = 13305
    lemonade_build = args.lemonade_build_dir.resolve()
    lemonade = lemonade_build / "lemonade"
    llama_server = args.llama_server.resolve()
    models_dir = (
        args.models_dir.resolve()
        if args.models_dir is not None
        else None
    )
    phases = (
        BenchmarkPhase(
            name="HRX",
            backend="hrx",
            device="HRX0",
            output=args.hrx_output,
            server_log=args.hrx_server_log,
            response_log=args.hrx_response_log,
        ),
        BenchmarkPhase(
            name="Vulkan",
            backend="vulkan",
            device="Vulkan0",
            output=args.vulkan_output,
            server_log=args.vulkan_server_log,
            response_log=args.vulkan_response_log,
        ),
    )
    failed_scenario_count = 0

    try:
        # Precreate debug files so early failures still leave uploadable artifacts.
        for log_path in (
            args.hrx_server_log,
            args.vulkan_server_log,
            args.hrx_response_log,
            args.vulkan_response_log,
        ):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if args.batched:
                log_path.touch(exist_ok=True)
            else:
                log_path.write_text("", encoding="utf-8")

        cache_dir, hf_home, runtime_dir = prepare_state(args.state_root)
        env = dict(os.environ)
        env.update(
            {
                "HF_HOME": os.fspath(hf_home),
                "HF_HUB_CACHE": os.fspath(hf_home / "hub"),
                "LEMONADE_CACHE_DIR": os.fspath(cache_dir),
                "LEMONADE_HOST": "127.0.0.1",
                "LEMONADE_PORT": str(port),
                "XDG_RUNTIME_DIR": os.fspath(runtime_dir),
            }
        )

        log_llama_server_devices(llama_server, env)

        # Run benchmarks
        for phase_index, phase in enumerate(phases):
            log(
                f"Starting {phase.name} benchmark phase on {phase.device}"
            )
            active_phase = phase
            if args.batched:
                assert args.batch_number is not None
                batch_root = args.state_root / phase.backend
                active_phase = BenchmarkPhase(
                    name=phase.name,
                    backend=phase.backend,
                    device=phase.device,
                    output=batch_root / "benchmark.json",
                    server_log=batch_root / "server.log",
                    response_log=batch_root / "responses.jsonl",
                )

            server_process: subprocess.Popen[str] | None = None
            try:
                server_process = start_lemond(
                    lemonade_build,
                    cache_dir,
                    active_phase.server_log,
                    env=env,
                    port=port,
                )
                wait_for_live(server_process, port)
                set_lemonade_config_values(
                    lemonade,
                    llama_server,
                    phase.device,
                    models_dir=models_dir,
                    env=env,
                    port=port,
                )
                batch_data, phase_failed_scenario_count = run_benchmark(
                    lemonade,
                    phase.backend,
                    active_phase.output,
                    active_phase.response_log,
                    args.models,
                    auto_pull=models_dir is None and phase_index == 0,
                    env=env,
                )
            finally:
                try:
                    server_return_code = stop_server(server_process)
                    log(
                        f"{phase.name} lemond stopped with status "
                        f"{server_return_code}"
                    )
                    if (
                        server_process is not None
                        and server_return_code != 0
                    ):
                        raise RuntimeError(
                            f"{phase.name} lemond stopped unexpectedly with "
                            f"status {server_return_code}"
                        )
                finally:
                    if args.batched:
                        assert args.batch_number is not None
                        for destination, source in (
                            (phase.server_log, active_phase.server_log),
                            (phase.response_log, active_phase.response_log),
                        ):
                            try:
                                append_batch_log(
                                    destination,
                                    source,
                                    args.batch_number,
                                )
                            except OSError as exc:
                                log(
                                    "Warning: could not append "
                                    f"{phase.backend} batch {args.batch_number} "
                                    f"log to {destination}: {exc}"
                                )

            verify_configured_executable(
                phase=active_phase,
                phases=phases,
                cache_dir=cache_dir,
                llama_server=llama_server,
            )
            if args.batched:
                merged_count = merge_benchmark_output(
                    phase.output,
                    batch_data,
                )
                log(
                    f"Merged {merged_count} {phase.backend} model(s) from "
                    f"batch {args.batch_number} into {phase.output}"
                )
            failed_scenario_count += phase_failed_scenario_count
    finally:
        remove_state_root(args.state_root)
        log(f"Cleaned benchmark state: {args.state_root}")

    if failed_scenario_count:
        log(
            "Lemonade benchmark completed with "
            f"{failed_scenario_count} failed scenario(s)"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lemonade-build-dir", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--hrx-output", type=Path, required=True)
    parser.add_argument("--vulkan-output", type=Path, required=True)
    parser.add_argument("--hrx-server-log", type=Path, required=True)
    parser.add_argument("--vulkan-server-log", type=Path, required=True)
    parser.add_argument("--hrx-response-log", type=Path, required=True)
    parser.add_argument("--vulkan-response-log", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    if args.batched and args.batch_number is None:
        parser.error("--batch-number is required with --batched")
    if args.batch_number is not None and not args.batched:
        parser.error("--batch-number requires --batched")
    if args.batch_number is not None and args.batch_number < 1:
        parser.error("--batch-number must be a positive integer")
    if args.state_root is None:
        if not args.batched:
            parser.error("--state-root is required without --batched")
        if args.models_dir is None:
            parser.error("--models-dir is required when --state-root is omitted")
        args.state_root = args.models_dir.parent / "lemonade-state"
    if args.models_dir is not None:
        try:
            state_root = args.state_root.resolve()
            models_dir = args.models_dir.resolve()
        except (OSError, RuntimeError) as exc:
            parser.error(f"Could not resolve state or model path: {exc}")
        paths_are_equal = state_root == models_dir
        state_contains_models = state_root in models_dir.parents
        models_contain_state = models_dir in state_root.parents
        paths_overlap = (
            paths_are_equal
            or state_contains_models
            or models_contain_state
        )
        if paths_overlap:
            parser.error("--state-root and --models-dir must not overlap")
    try:
        return run(args)
    except (
        OSError,
        RuntimeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Lemonade benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
