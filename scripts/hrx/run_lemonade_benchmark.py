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


@dataclass
class BenchmarkPhase:
    name: str
    backend: str
    device: str
    output: Path
    server_log: Path
    response_log: Path


class BenchmarkValidationError(RuntimeError):
    """Raised when a benchmark scenario reports failed runs."""


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


def validate_benchmark(data: dict[str, Any]) -> int:
    """Fail when any scenario reports one or more failed runs."""
    scenario_count = 0
    for model in data["models"]:
        for result in model["results"]:
            for scenario in result["scenarios"]:
                failed_runs = scenario["failed_runs"]
                if failed_runs != 0:
                    raise BenchmarkValidationError(
                        f"{model['model']!r} scenario {scenario['name']!r} reported "
                        f"{failed_runs} failed run(s)"
                    )
                scenario_count += 1
    return scenario_count


def run_benchmark(
    executable: Path,
    backend: str,
    output: Path,
    response_log: Path,
    models: list[str],
    *,
    auto_pull: bool,
    env: dict[str, str],
) -> None:
    """Run and validate one backend's complete benchmark suite."""
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
        "--llamacpp-args=--ignore-eos",
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
    scenario_count = validate_benchmark(data)
    log(
        f"Validated {scenario_count} {backend} scenario(s) with zero failed "
        f"runs in {output}"
    )


def set_lemonade_config_values(
    executable: Path,
    llama_server: Path,
    device: str,
    *,
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
    log("++ " + " ".join(command))
    subprocess.run(command, env=env, check=True)
    config = request_json(port, "/internal/config")
    if (
        Path(config["llamacpp"]["hrx_bin"]).resolve() != llama_server
        or Path(config["llamacpp"]["vulkan_bin"]).resolve() != llama_server
        or config["llamacpp"]["device"] != device
        or config["no_fetch_executables"] is not True
        or config["log_level"] != "debug"
    ):
        raise RuntimeError("Lemonade did not retain the requested configuration")
    log(
        "Verified Lemonade configuration: "
        f"llamacpp.hrx_bin={llama_server}, "
        f"llamacpp.vulkan_bin={llama_server}, "
        f"llamacpp.device={device}, no_fetch_executables=true, log_level=debug"
    )


def verify_configured_executable(
    *,
    phases: tuple[BenchmarkPhase, ...],
    phase_index: int,
    cache_dir: Path,
    llama_server: Path,
) -> None:
    """Verify log and cache evidence for the pinned binary and selected device."""
    phase = phases[phase_index]
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
        for index, other_phase in enumerate(phases)
        if index != phase_index
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

    try:
        # Precreate debug files so early failures still leave uploadable artifacts.
        for log_path in (
            args.hrx_server_log,
            args.vulkan_server_log,
            args.hrx_response_log,
            args.vulkan_response_log,
        ):
            log_path.parent.mkdir(parents=True, exist_ok=True)
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
            server_process: subprocess.Popen[str] | None = None
            try:
                server_process = start_lemond(
                    lemonade_build,
                    cache_dir,
                    phase.server_log,
                    env=env,
                    port=port,
                )
                wait_for_live(server_process, port)
                set_lemonade_config_values(
                    lemonade,
                    llama_server,
                    phase.device,
                    env=env,
                    port=port,
                )
                run_benchmark(
                    lemonade,
                    phase.backend,
                    phase.output,
                    phase.response_log,
                    args.models,
                    auto_pull=phase_index == 0,
                    env=env,
                )
            finally:
                server_return_code = stop_server(server_process)
                log(
                    f"{phase.name} lemond stopped with status {server_return_code}"
                )

            verify_configured_executable(
                phases=phases,
                phase_index=phase_index,
                cache_dir=cache_dir,
                llama_server=llama_server,
            )
    finally:
        remove_state_root(args.state_root)
        log(f"Cleaned benchmark state: {args.state_root}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lemonade-build-dir", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--hrx-output", type=Path, required=True)
    parser.add_argument("--vulkan-output", type=Path, required=True)
    parser.add_argument("--hrx-server-log", type=Path, required=True)
    parser.add_argument("--vulkan-server-log", type=Path, required=True)
    parser.add_argument("--hrx-response-log", type=Path, required=True)
    parser.add_argument("--vulkan-response-log", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
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
