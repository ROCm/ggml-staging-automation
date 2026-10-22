#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Benchmark identical pinned GGUFs through released Lemonade's HRX/Vulkan recipes.

The manifest owns model identity, optional MTP cases and per-case expectations.
Private user registrations share the verified files through a pinned HF snapshot;
no model or executable downloads are allowed. Each case uses ordinary Lemonade
bench, including natural EOS. A local relay captures its existing completion
responses because bench's own artifact omits draft counters. Counters include
warmups and are report-only; an on-case must actually draft, and off must not.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmark_output import append_batch_log, atomic_write_json, merge_benchmark_output
from lemonade_capture import capture_responses, draft_metrics
from run_batched_benchmark import load_manifest

SCENARIOS = ("chat-short", "chat-long-output")


def log(message: str) -> None:
    print(message, flush=True)


def prepare_state(state_root: Path) -> tuple[Path, Path, Path]:
    """Keep Lemonade runtime state under one private temporary root."""
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
    lemonade_dir: Path,
    cache_dir: Path,
    log_path: Path,
    env: dict[str, str],
    port: int,
) -> subprocess.Popen[str]:
    """Start the daemon that owns Lemonade configuration and backend processes."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.fspath(lemonade_dir / "lemond"),
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


def configure(executable, llama_server, env, port):
    settings = {
        "hrx.hrx_bin": str(llama_server),
        "llamacpp.vulkan_bin": str(llama_server),
        "no_fetch_executables": "true", "offline": "true", "log_level": "debug",
    }
    command = [str(executable), "--host", "127.0.0.1", "--port", str(port),
               "config", "set", *(f"{key}={value}" for key, value in settings.items())]
    subprocess.run(command, env=env, check=True)
    config = request_json(port, "/internal/config")
    for section, key in (("hrx", "hrx_bin"), ("llamacpp", "vulkan_bin")):
        if Path(config[section][key]).resolve() != llama_server:
            raise RuntimeError(f"Lemonade did not retain {section}.{key}")
    for key in ("no_fetch_executables", "offline"):
        if config[key] is not True:
            raise RuntimeError(f"Lemonade did not retain {key}")


def register_models(models, models_dir, cache_dir, hf_home):
    definitions = {}
    options = {}
    for model in models:
        snapshot = hf_home / "hub" / ("models--" + model.repository.replace("/", "--"))
        target = snapshot / "snapshots" / model.revision / model.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((models_dir / model.directory / model.filename).resolve(strict=True))
        refs = snapshot / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(model.revision, encoding="utf-8")
        for backend, recipe in (("hrx", "llamacpp-hrx"), ("vulkan", "llamacpp")):
            for mode in model.mtp:
                name = registration_name(model, backend, mode)
                definitions[name] = {"recipe": recipe, "labels": ["chat", "custom"],
                    "checkpoint": f"{model.repository}:{model.filename}"}
                argument_key = "hrx_args" if backend == "hrx" else "llamacpp_args"
                # Only disable explicitly. On must exercise Lemonade's auto-enablement.
                arguments = "--spec-type none" if mode == "off" else ""
                options["user." + name] = {argument_key: arguments, "merge_args": True}
                if backend == "vulkan":
                    options["user." + name]["llamacpp_device"] = "Vulkan0"
    atomic_write_json(cache_dir / "user_models.json", definitions)
    atomic_write_json(cache_dir / "recipe_options.json", options)


def registration_name(model, backend, mode):
    return f"ci-{model.id}-{backend}-{mode}"


def classify(expectation, succeeded):
    if expectation == "report-only":
        return "REPORT_ONLY"
    if expectation == "fail":
        return "XPASS" if succeeded else "XFAIL"
    return "PASS" if succeeded else "FAIL"


def run_case(executable, model, backend, mode, root, env, port, response_log):
    name = registration_name(model, backend, mode)
    output = root / f"{name}.json"
    with capture_responses(port, response_log, name) as (capture_port, records):
        command = [str(executable), "--host", "127.0.0.1", "--port", str(capture_port),
                   "bench", "--backend", backend, "--ctx-size", "4096",
                   "--warmup", "1", "--runs", "3", "--scenarios", SCENARIOS[0],
                   "--scenarios", SCENARIOS[1], "--output", str(output), name]
        log("++ " + " ".join(command))
        result = subprocess.run(command, env=env, check=False)
    metrics = draft_metrics(records)
    errors = []
    if result.returncode:
        errors.append(f"lemonade bench exited {result.returncode}")
    document = None
    recipe = "llamacpp-hrx" if backend == "hrx" else "llamacpp"
    if output.exists():
        document = json.loads(output.read_text(encoding="utf-8"))
        rows = document["models"]
        if len(rows) != 1:
            raise RuntimeError("Expected one model from single-case benchmark")
        row = rows[0]
        if row["model"] not in (name, "user." + name):
            raise RuntimeError("Lemonade returned the wrong registered model")
        results = row["results"]
        if len(results) != 1:
            raise RuntimeError("Expected one configuration from single-case benchmark")
        measured = results[0]
        matching_backend = measured["backend"] == backend
        matching_recipe = measured["recipe"] == recipe
        correct_route = matching_backend and matching_recipe
        if not correct_route:
            raise RuntimeError("Lemonade benchmark used the wrong backend/recipe")
        scenarios = measured["scenarios"]
        if [s["name"] for s in scenarios] != list(SCENARIOS):
            raise RuntimeError("Lemonade returned unexpected scenarios")
        for scenario in scenarios:
            failed = scenario["failed_runs"]
            integer_count = type(failed) is int
            if not integer_count:
                raise RuntimeError("Non-integer failed-run count from Lemonade")
            if failed < 0:
                raise RuntimeError("Invalid failed-run count from Lemonade")
            has_failed_runs = failed > 0
            all_runs_failed = scenario.get("all_runs_failed", False)
            scenario_failed = has_failed_runs or all_runs_failed
            if scenario_failed:
                errors.append(f"{scenario['name']}: inference failed")
    else:
        raise RuntimeError(
            f"Lemonade exited {result.returncode} without a benchmark artifact for {name}"
        )
    requires_drafting = mode == "on"
    has_drafts = metrics["drafted"] > 0
    missing_drafts = requires_drafting and not has_drafts
    forbidden_drafts = mode == "off" and has_drafts
    if missing_drafts:
        errors.append("MTP requested but no drafted tokens observed")
    if forbidden_drafts:
        errors.append("MTP off produced drafted tokens")
    expectation = model.expectation("lemonade-benchmark", backend, mode)
    succeeded = not errors
    outcome = classify(expectation, succeeded)
    label = model.name if mode == "default" else f"{model.name} [MTP {mode}]"
    row.update(model=label, model_id=model.id, mtp=mode,
               registration="user." + name, expected_result=expectation,
               result="pass" if succeeded else "fail", outcome=outcome,
               speculative=metrics, errors=errors,
               weights={"revision": model.revision, "sha256": model.sha256})
    log(f"{outcome}: {backend} {label}: {metrics}; {errors}")
    return row, document, outcome in ("FAIL", "XPASS")


def verify_executable(log_path, cache_dir, llama_server, backend):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Verify the external binary selected by offline backend installation and
    # every logged launch. Preflight/config failures cannot be model XFAILs.
    recipe = "llamacpp-hrx" if backend == "hrx" else "llamacpp"
    marker = f"; using installed {recipe}:{backend} backend at {llama_server}"
    if marker not in text:
        raise RuntimeError(f"No evidence of configured {backend} executable")
    device = "HRX0" if backend == "hrx" else "Vulkan0"
    for line in text.splitlines():
        if "Starting process with filtered output: " in line:
            command = line.split("Starting process with filtered output: ", 1)[1]
            if not command.startswith(str(llama_server) + " "):
                raise RuntimeError(f"Unexpected backend executable: {command}")
            if f"--device {device}" not in command:
                raise RuntimeError(f"Wrong {backend} device: {command}")
    # Lemonade creates the empty bin root even when fetching is disabled.
    cached_files = [path for path in (cache_dir / "bin").rglob("*") if path.is_file()]
    if cached_files:
        raise RuntimeError(f"Unexpected managed executable files: {cached_files}")


def run(args):
    executable_dir = args.lemonade_dir.resolve()
    llama_server = args.llama_server.resolve()
    models_dir = args.models_dir.resolve()
    manifest = load_manifest(args.model_manifest)
    by_name = {model.name: model for model in manifest.models}
    models = [by_name[name] for name in args.models]
    unexpected = False
    for backend in ("hrx", "vulkan"):
        destination = getattr(args, backend + "_output")
        server_log = getattr(args, backend + "_server_log")
        response_log = getattr(args, backend + "_response_log")
        with tempfile.TemporaryDirectory(dir=models_dir.parent, prefix="lemonade-state-") as tmp:
            root = Path(tmp)
            cache_dir, hf_home, runtime_dir = prepare_state(root)
            register_models(models, models_dir, cache_dir, hf_home)
            env = dict(os.environ, HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home / "hub"),
                       LEMONADE_CACHE_DIR=str(cache_dir), XDG_RUNTIME_DIR=str(runtime_dir))
            log_path = root / "server.log"
            captured = root / "responses.jsonl"
            captured.touch()
            process = None
            data = {"models": []}
            try:
                process = start_lemond(executable_dir, cache_dir, log_path, env, args.port)
                wait_for_live(process, args.port)
                configure(executable_dir / "lemonade", llama_server, env, args.port)
                for model in models:
                    for mode in model.mtp:
                        row, document, failed = run_case(executable_dir / "lemonade", model,
                            backend, mode, root, env, args.port, captured)
                        if document is not None:
                            data.update({k: v for k, v in document.items() if k != "models"})
                        data["models"].append(row)
                        unexpected = unexpected or failed
                verify_executable(log_path, cache_dir, llama_server, backend)
            finally:
                daemon_returncode = stop_server(process)
                if args.batched:
                    merge_benchmark_output(destination, data)
                else:
                    atomic_write_json(destination, data)
                for target, source in ((server_log, log_path), (response_log, captured)):
                    if not args.batched:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text("", encoding="utf-8")
                    append_batch_log(target, source, args.batch_number or 1)
                daemon_started = process is not None
                daemon_failed = daemon_returncode != 0
                unexpected_shutdown = daemon_started and daemon_failed
                if unexpected_shutdown:
                    raise RuntimeError(f"{backend} daemon exited {daemon_returncode}")
    return int(unexpected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("lemonade-dir", "llama-server", "models-dir", "model-manifest",
                   "hrx-output", "vulkan-output", "hrx-server-log", "vulkan-server-log",
                   "hrx-response-log", "vulkan-response-log"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--port", type=int, default=13305)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch-number", type=int)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Lemonade benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
