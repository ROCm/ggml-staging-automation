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


def initialize_outputs(args: argparse.Namespace) -> None:
    metadata = {
        "schema_version": 1,
        "llama_cpp": args.llama_cpp_commit,
        "hrx_system": args.hrx_system_commit,
        "lemonade": args.lemonade_commit,
    }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for log_path in (args.server_log, args.benchmark_log):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    print(f"Wrote benchmark metadata to {args.metadata_output}", flush=True)


def remove_state_root(state_root: Path) -> None:
    if state_root.exists():
        shutil.rmtree(state_root)


def prepare_state(state_root: Path) -> dict[str, Path]:
    remove_state_root(state_root)
    cache_dir = state_root / "lemonade"
    hf_home = state_root / "huggingface"
    runtime_dir = state_root / "runtime"
    for path in (cache_dir, hf_home, runtime_dir):
        path.mkdir(parents=True)
    runtime_dir.chmod(0o700)
    return {
        "cache": cache_dir,
        "hf_home": hf_home,
        "runtime": runtime_dir,
    }


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


def run_captured(
    command: list[str], *, env: dict[str, str], description: str
) -> None:
    print("++", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with status {result.returncode}")


def stream_benchmark(
    command: list[str], *, env: dict[str, str], log_path: Path
) -> int:
    print("++", " ".join(command), flush=True)
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    return return_code


def run_benchmark(
    executable: Path,
    output: Path,
    *,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, str, bool]:
    model = "Qwen3-30B-A3B-Instruct-2507-GGUF"
    backend = "hrx"
    command = [
        os.fspath(executable),
        "bench",
        "--backend",
        backend,
        "--auto-pull",
        "--output",
        os.fspath(output),
        model,
    ]
    return_code = stream_benchmark(command, env=env, log_path=log_path)
    if return_code != 0 or not output.is_file():
        return return_code, "", False

    table = format_markdown_results(output)
    return return_code, table, append_summary(table)


def format_markdown_results(benchmark_json: Path) -> str:
    data = json.loads(benchmark_json.read_text(encoding="utf-8"))
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


def append_summary(table: str) -> bool:
    summary_value = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_value:
        return False
    summary_path = Path(summary_value)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write(table)
        summary.write("\n")
    return True


def verify_config(config: dict[str, Any], llama_server: Path) -> None:
    if (
        Path(config["llamacpp"]["hrx_bin"]).resolve() != llama_server
        or config["no_fetch_executables"] is not True
    ):
        raise RuntimeError("Lemonade did not retain the requested configuration")


def set_lemonade_config_values(
    executable: Path,
    llama_server: Path,
    env: dict[str, str],
    port: int,
) -> None:
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
    run_captured(command, env=env, description="Lemonade configuration")
    config = request_json(port, "/internal/config")
    verify_config(config, llama_server)
    print(
        "Verified Lemonade configuration: "
        f"llamacpp.hrx_bin={llama_server}, "
        "no_fetch_executables=true",
        flush=True,
    )


def used_configured_executable(
    *, server_log: Path, state: dict[str, Path], llama_server: Path
) -> bool:
    forbidden_install_signature = "Installing llama-server"
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    positive_text = (
        "Fetching executable artifacts is disabled; using installed "
        f"llamacpp:hrx backend at {llama_server}"
    )
    managed_hrx_dir = state["cache"] / "bin" / "llamacpp" / "hrx"
    return (
        positive_text in log_text
        and not managed_hrx_dir.exists()
        and forbidden_install_signature not in log_text
    )


def run(args: argparse.Namespace) -> int:
    port = 13305
    server_process: subprocess.Popen[str] | None = None
    state: dict[str, Path] | None = None
    benchmark_return_code: int | None = None
    no_download = False
    summary_written = False
    table = ""

    initialize_outputs(args)

    try:
        state = prepare_state(args.state_root)

        lemonade_build = args.lemonade_build_dir.resolve()
        lemonade = lemonade_build / "lemonade"
        llama_server = args.llama_server.resolve()
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

        env = dict(os.environ)
        env.update(
            {
                "HF_HOME": os.fspath(state["hf_home"]),
                "HF_HUB_CACHE": os.fspath(state["hf_home"] / "hub"),
                "LEMONADE_CACHE_DIR": os.fspath(state["cache"]),
                "LEMONADE_HOST": "127.0.0.1",
                "LEMONADE_PORT": str(port),
                "XDG_RUNTIME_DIR": os.fspath(state["runtime"]),
            }
        )

        server_process = start_lemond(
            lemonade_build,
            state["cache"],
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

        benchmark_return_code, table, summary_written = run_benchmark(
            lemonade,
            output,
            env=env,
            log_path=args.benchmark_log,
        )
    finally:
        server_return_code = stop_server(server_process)
        print(f"lemond stopped with status {server_return_code}", flush=True)
        if state is not None and server_process is not None:
            no_download = used_configured_executable(
                server_log=args.server_log,
                state=state,
                llama_server=args.llama_server.resolve(),
            )

    if benchmark_return_code != 0:
        raise RuntimeError(
            f"Lemonade benchmark failed with status {benchmark_return_code}"
        )
    if not args.output.is_file():
        raise RuntimeError("Lemonade benchmark did not write benchmark.json")
    if not table:
        raise RuntimeError("Lemonade benchmark did not produce its summary table")
    if os.environ.get("GITHUB_ACTIONS") == "true" and not summary_written:
        raise RuntimeError("Lemonade benchmark table was not written to the job summary")
    if not no_download:
        raise RuntimeError(
            "Could not prove that Lemonade used the configured HRX executable "
            "without installing another backend"
        )

    print(
        "Verified Lemonade used the extracted llama-server without installing "
        "another backend.",
        flush=True,
    )
    return 0


def cleanup(args: argparse.Namespace) -> int:
    remove_state_root(args.state_root)
    print(f"Cleaned benchmark state: {args.state_root}", flush=True)
    return 0


def add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run", help="initialize outputs and run the isolated Lemonade benchmark"
    )
    parser.add_argument("--lemonade-build-dir", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    parser.add_argument("--hrx-system-commit", required=True)
    parser.add_argument("--lemonade-commit", required=True)
    parser.set_defaults(handler=run)


def add_cleanup_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("cleanup", help="remove isolated benchmark state")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.set_defaults(handler=cleanup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    add_cleanup_parser(subparsers)
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Lemonade benchmark {args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
