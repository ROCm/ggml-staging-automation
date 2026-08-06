#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the release benchmark through an isolated Lemonade daemon."""

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
from pathlib import Path
from typing import Any


class BenchmarkValidationError(RuntimeError):
    """Raised when a benchmark does not match the required result schema."""


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
    print("++", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def require_nonempty_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{location} must be a nonempty array")
    return value


def validate_benchmark(data: object) -> int:
    if not isinstance(data, dict):
        raise BenchmarkValidationError("benchmark root must be an object")

    models = require_nonempty_list(data.get("models"), "models")
    scenario_count = 0
    for model_index, model in enumerate(models):
        model_location = f"models[{model_index}]"
        if not isinstance(model, dict):
            raise BenchmarkValidationError(f"{model_location} must be an object")

        results = require_nonempty_list(
            model.get("results"), f"{model_location}.results"
        )
        for result_index, result in enumerate(results):
            result_location = f"{model_location}.results[{result_index}]"
            if not isinstance(result, dict):
                raise BenchmarkValidationError(f"{result_location} must be an object")

            scenarios = require_nonempty_list(
                result.get("scenarios"), f"{result_location}.scenarios"
            )
            for scenario_index, scenario in enumerate(scenarios):
                scenario_location = f"{result_location}.scenarios[{scenario_index}]"
                if not isinstance(scenario, dict):
                    raise BenchmarkValidationError(
                        f"{scenario_location} must be an object"
                    )

                name = scenario.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise BenchmarkValidationError(
                        f"{scenario_location}.name must be a nonempty string"
                    )

                failed_runs = scenario.get("failed_runs")
                if isinstance(failed_runs, bool) or not isinstance(failed_runs, int):
                    raise BenchmarkValidationError(
                        f"{scenario_location}.failed_runs must be an integer"
                    )
                if failed_runs != 0:
                    raise BenchmarkValidationError(
                        f"{scenario_location} ({name!r}) reported "
                        f"{failed_runs} failed run(s)"
                    )
                if scenario.get("all_runs_failed") is True:
                    raise BenchmarkValidationError(
                        f"{scenario_location} ({name!r}) reports that all runs failed"
                    )
                scenario_count += 1

    return scenario_count


def run_benchmark(
    executable: Path,
    output: Path,
    models: list[str],
    *,
    env: dict[str, str],
) -> None:
    """Run the user-facing HRX workload and reject invalid results before upload."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [
        os.fspath(executable),
        "bench",
        "--backend",
        "hrx",
        "--auto-pull",
        "--output",
        os.fspath(output),
        *models,
    ]
    print("++", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    if not output.is_file():
        raise RuntimeError("Lemonade benchmark did not write benchmark.json")

    data = json.loads(output.read_text(encoding="utf-8"))
    scenario_count = validate_benchmark(data)
    print(
        f"Validated {scenario_count} successful Lemonade benchmark scenario(s) "
        f"in {output}",
        flush=True,
    )
    append_summary(format_markdown_results(data))


def format_markdown_results(data: dict[str, Any]) -> str:
    lines = ["## Lemonade benchmark results", ""]

    for model in data["models"]:
        lines.extend([f"### `{model['model']}`", ""])
        for result in model["results"]:
            backend = f"{result['recipe']}/{result['backend']}"
            lines.extend(
                [
                    f"**Backend:** `{backend}` · "
                    f"**Context:** `{result['ctx_size']}` tokens",
                    "",
                    "| Scenario | TTFT mean (ms) | TTFT min (ms) | "
                    "TTFT max (ms) | TPS mean | TPS min | TPS max | "
                    "VRAM peak (GB) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for scenario in result["scenarios"]:
                ttft = scenario["ttft_ms"]
                tps = scenario["tps"]
                lines.append(
                    f"| {scenario['name']} | {ttft['mean']:.1f} | "
                    f"{ttft['min']:.1f} | {ttft['max']:.1f} | "
                    f"{tps['mean']:.1f} | {tps['min']:.1f} | "
                    f"{tps['max']:.1f} | {scenario['vram_peak_gb']:.1f} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip()


def append_summary(table: str) -> None:
    """Publish readable results and fail CI if Actions cannot record the summary."""
    summary_value = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_value:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            raise RuntimeError(
                "Lemonade benchmark table was not written to the job summary"
            )
        return
    with Path(summary_value).open("a", encoding="utf-8") as summary:
        summary.write(table)
        summary.write("\n")


def set_lemonade_config_values(
    executable: Path,
    llama_server: Path,
    env: dict[str, str],
    port: int,
) -> None:
    """Pin the extracted llama-server and prevent fallback executable downloads."""
    command = [
        os.fspath(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "config",
        "set",
        f"llamacpp.hrx_bin={llama_server}",
        "no_fetch_executables=true",
    ]
    print("++", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    config = request_json(port, "/internal/config")
    if (
        Path(config["llamacpp"]["hrx_bin"]).resolve() != llama_server
        or config["no_fetch_executables"] is not True
    ):
        raise RuntimeError("Lemonade did not retain the requested configuration")
    print(
        "Verified Lemonade configuration: "
        f"llamacpp.hrx_bin={llama_server}, "
        "no_fetch_executables=true",
        flush=True,
    )


def verify_configured_executable(
    *, server_log: Path, cache_dir: Path, llama_server: Path
) -> None:
    """Confirm Lemonade honored the pinned executable before deleting its state."""
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    positive_text = (
        "Fetching executable artifacts is disabled; using installed "
        f"llamacpp:hrx backend at {llama_server}"
    )
    managed_hrx_dir = cache_dir / "bin" / "llamacpp" / "hrx"
    if (
        positive_text not in log_text
        or managed_hrx_dir.exists()
        or "Installing llama-server" in log_text
    ):
        raise RuntimeError(
            "Could not prove that Lemonade used the configured HRX executable "
            "without installing another backend"
        )
    print(
        "Verified Lemonade used the extracted llama-server without installing "
        "another backend.",
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    port = 13305
    server_process: subprocess.Popen[str] | None = None
    benchmark_succeeded = False

    try:
        # Precreate the file so early startup failures still have a debug artifact.
        args.server_log.parent.mkdir(parents=True, exist_ok=True)
        args.server_log.write_text("", encoding="utf-8")
        cache_dir, hf_home, runtime_dir = prepare_state(args.state_root)

        lemonade_build = args.lemonade_build_dir.resolve()
        lemonade = lemonade_build / "lemonade"
        llama_server = args.llama_server.resolve()
        output = args.output.resolve()

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

        server_process = start_lemond(
            lemonade_build,
            cache_dir,
            args.server_log.resolve(),
            env=env,
            port=port,
        )
        wait_for_live(server_process, port)

        set_lemonade_config_values(
            lemonade,
            llama_server,
            env=env,
            port=port,
        )

        run_benchmark(
            lemonade,
            output,
            args.models,
            env=env,
        )
        benchmark_succeeded = True
    finally:
        try:
            server_return_code = stop_server(server_process)
            print(f"lemond stopped with status {server_return_code}", flush=True)
            if benchmark_succeeded:
                verify_configured_executable(
                    server_log=args.server_log,
                    cache_dir=cache_dir,
                    llama_server=llama_server,
                )
        finally:
            remove_state_root(args.state_root)
            print(f"Cleaned benchmark state: {args.state_root}", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lemonade-build-dir", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
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
