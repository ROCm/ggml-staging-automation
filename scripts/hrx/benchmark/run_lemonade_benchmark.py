#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Benchmark identical pinned GGUFs through released Lemonade's HRX/Vulkan recipes.

The manifest owns model identity, optional MTP cases and per-case expectations.
Private user registrations point directly to the same verified local GGUF;
no model or executable downloads are allowed. Each case uses ordinary Lemonade
bench, including natural EOS. MTP on uses Lemonade's automatic defaults;
verification of actual drafting is deferred until counters are readily available.
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
from typing import Any

from benchmark_output import append_batch_log, atomic_write_json, merge_benchmark_output
from run_batched_benchmark import load_manifest

SCENARIOS = ("chat-short", "chat-long-output")


def log(message: str) -> None:
    print(message, flush=True)


def prepare_state(state_root: Path) -> tuple[Path, Path]:
    """Create the private cache and runtime directories for one backend run.

    The cache is passed both as lemond's positional state directory and as
    LEMONADE_CACHE_DIR for its CLI clients. It holds our user_models.json,
    recipe_options.json and daemon configuration, isolating the run from the
    runner's normal Lemonade installation. XDG_RUNTIME_DIR supplies a private,
    owner-only runtime directory. Model weights remain outside this state tree.
    """
    cache_dir = state_root / "lemonade"
    runtime_dir = state_root / "runtime"
    for path in (cache_dir, runtime_dir):
        path.mkdir(parents=True)
    runtime_dir.chmod(0o700)
    return cache_dir, runtime_dir


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
    """Wait for lemond's /live endpoint before sending configuration commands.

    This checks the Lemonade control service, not a loaded llama-server model.
    Model loading happens later through `lemonade bench`; readiness here only
    means the daemon can accept the setup requests.
    """
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
    """Let lemond shut down its backend before removing the private state.

    SIGTERM gives Lemonade a chance to stop the llama-server process it owns
    and flush its logs. If shutdown stalls, signal the process group created
    by start_lemond so backend children cannot retain the GPU after this run.
    Return the daemon's exit status so cleanup cannot hide an unexpected exit.
    """
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
    """Launch the released embeddable lemond with our prepared state directory.

    lemond reads the user registrations and recipe options written before
    startup, serves the CLI's HTTP requests, and owns backend installation and
    process launches. Explicit host/port arguments direct our CLI calls to
    this loopback service. A new process session permits group cleanup if the
    daemon or its backend fails to stop normally; debug output is retained for
    executable/device verification and CI diagnostics.
    """
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
    """Select the CI executable through Lemonade's backend configuration keys.

    `lemonade config set` updates the running daemon: hrx.hrx_bin supplies the
    executable for the llamacpp-hrx recipe, while llamacpp.vulkan_bin supplies
    the Vulkan backend of the regular llamacpp recipe. Both point to the same
    CI-built llama-server; recipe/device selection chooses which backend runs.

    offline disables network-dependent operation and no_fetch_executables
    prevents Lemonade from installing a managed backend in place of our build.
    Read /internal/config back to verify the effective settings, and enable
    debug logging so verify_executable can check the subsequent selection and
    launch records as well as this configuration intent.
    """
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


def register_models(models, models_dir, cache_dir):
    """Prepare Lemonade user registrations and per-model recipe overrides.

    user_models.json stores bare registration names; Lemonade exposes these
    in its user namespace. Each backend/mode gets its own registration so its
    recipe options and benchmark results are independent. The llamacpp-hrx
    and regular llamacpp registrations use source=local_path with the same
    absolute GGUF checkpoint. Lemonade's model resolver reads that file
    directly, bypassing Hugging Face lookup or download. The batch runner has
    already verified the weights against the manifest's checksum.

    recipe_options.json is keyed by canonical user.<name> IDs. hrx_args and
    llamacpp_args extend the respective recipe's arguments with merge_args
    enabled, preserving Lemonade's defaults. Both disable server prompt reuse
    so measured runs process the full prompt while sharing warmup's compiled
    kernels. Off explicitly adds --spec-type none; on/default leave automatic
    MTP selection to Lemonade. This configures intent only; actual drafting is
    not currently verified.

    Vulkan uses the managed llamacpp_device option to select Vulkan0, since
    device selection belongs to Lemonade rather than arbitrary backend args.
    The HRX recipe selects HRX0 itself. Write both files before daemon startup
    so its initial registry sees all cases without a live registration update.
    """
    definitions = {}
    options = {}
    for model in models:
        checkpoint = str((models_dir / model.name / model.filename).resolve(strict=True))
        for backend, recipe in (("hrx", "llamacpp-hrx"), ("vulkan", "llamacpp")):
            for mode in model.modes("lemonade-benchmark"):
                name = registration_name(model, backend, mode)
                definitions[name] = {"recipe": recipe, "labels": ["chat", "custom"],
                    "source": "local_path", "checkpoint": checkpoint}
                argument_key = "hrx_args" if backend == "hrx" else "llamacpp_args"
                arguments = "--no-cache-prompt"
                if mode == "mtp:off":
                    arguments = "--spec-type none " + arguments
                options["user." + name] = {argument_key: arguments, "merge_args": True}
                if backend == "vulkan":
                    options["user." + name]["llamacpp_device"] = "Vulkan0"
    atomic_write_json(cache_dir / "user_models.json", definitions)
    atomic_write_json(cache_dir / "recipe_options.json", options)


def registration_name(model, backend, mode):
    """Return a bare user-model ID that separates backend and MTP options."""
    return f"ci-{model.id}-{backend}-{mode.replace(':', '-')}"


def classify(expectation, succeeded):
    """Apply the manifest's case expectation while keeping XPASS distinguishable."""
    if expectation == "report-only":
        return "REPORT_ONLY"
    if expectation == "fail":
        return "XPASS" if succeeded else "XFAIL"
    return "PASS" if succeeded else "FAIL"


def run_case(executable, model, backend, mode, root, env, port):
    """Run the ordinary Lemonade benchmark against one prepared registration.

    The CLI talks directly to our daemon, which loads the registered recipe
    and applies its per-model options. Explicit --backend selects HRX or Vulkan
    within that recipe. Lemonade owns warmup, measured requests and natural
    EOS behavior; this adapter consumes its JSON artifact rather than issuing
    extra generation requests.

    Check the artifact's model, recipe, backend and scenarios before assigning
    a case outcome. Missing or malformed artifacts are setup/protocol failures
    and cannot be hidden by a model's expected-failure policy. Valid results
    receive the manifest identity and requested MTP mode for report pairing;
    successful generation does not establish that speculative drafting ran.
    """
    name = registration_name(model, backend, mode)
    output = root / f"{name}.json"
    command = [str(executable), "--host", "127.0.0.1", "--port", str(port),
               "bench", "--backend", backend, "--ctx-size", "4096",
               "--no-reload", "--warmup", "1", "--runs", "3", "--scenarios", SCENARIOS[0],
               "--scenarios", SCENARIOS[1], "--output", str(output), name]
    log("++ " + " ".join(command))
    result = subprocess.run(command, env=env, check=False)
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
    expectation = model.expectation("lemonade-benchmark", backend, mode)
    succeeded = not errors
    outcome = classify(expectation, succeeded)
    label = model.name if mode == "default" else f"{model.name} [{mode}]"
    row.update(model=label, model_id=model.id, test_mode=mode,
               registration="user." + name, expected_result=expectation,
               result="pass" if succeeded else "fail", outcome=outcome,
               errors=errors,
               weights={"revision": model.revision, "sha256": model.sha256})
    log(f"{outcome}: {backend} {label}: {errors}")
    return row, document, outcome in ("FAIL", "XPASS")


def verify_executable(log_path, cache_dir, llama_server, backend):
    """Check Lemonade's backend-selection and process-launch debug records.

    The offline installer logs which externally configured executable it uses;
    require that marker for the expected recipe/backend. Check every recorded
    launch for the same CI binary and HRX0 or Vulkan0 device, then reject files
    in Lemonade's managed bin cache. This complements config readback with
    evidence of runtime selection. It relies on the pinned release's log
    format and does not establish that MTP actually drafted tokens.

    Failures here are infrastructure errors, outside model XFAIL handling.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
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
    """Run each backend in a fresh Lemonade daemon with private configuration.

    Registrations are prepared before lemond starts; its CLI then sets binary
    overrides before any model is loaded. Separate state and daemon lifetimes
    prevent one backend's loaded model or persisted options from influencing
    the other, while both consume the batch runner's same verified GGUFs.

    Preserve result artifacts and daemon logs in finally before deleting the
    temporary state, including partial evidence when setup or inference fails.
    Batched invocations merge into the workflow's accumulated outputs.
    """
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
        with tempfile.TemporaryDirectory(dir=models_dir.parent, prefix="lemonade-state-") as tmp:
            root = Path(tmp)
            cache_dir, runtime_dir = prepare_state(root)
            register_models(models, models_dir, cache_dir)
            env = dict(os.environ, LEMONADE_CACHE_DIR=str(cache_dir), XDG_RUNTIME_DIR=str(runtime_dir))
            log_path = root / "server.log"
            process = None
            data = {"models": []}
            try:
                process = start_lemond(executable_dir, cache_dir, log_path, env, args.port)
                wait_for_live(process, args.port)
                configure(executable_dir / "lemonade", llama_server, env, args.port)
                for model in models:
                    for mode in model.modes("lemonade-benchmark"):
                        row, document, failed = run_case(executable_dir / "lemonade", model,
                            backend, mode, root, env, args.port)
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
                if not args.batched:
                    server_log.parent.mkdir(parents=True, exist_ok=True)
                    server_log.write_text("", encoding="utf-8")
                append_batch_log(server_log, log_path, args.batch_number or 1)
                daemon_started = process is not None
                daemon_failed = daemon_returncode != 0
                unexpected_shutdown = daemon_started and daemon_failed
                if unexpected_shutdown:
                    raise RuntimeError(f"{backend} daemon exited {daemon_returncode}")
    return int(unexpected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("lemonade-dir", "llama-server", "models-dir", "model-manifest",
                   "hrx-output", "vulkan-output", "hrx-server-log", "vulkan-server-log"):
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
