#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bound device discovery and capture blocked thread stacks on PR 108 runners.

Reuse the exact experiment packages without loading a model. Compare default
backend discovery with Vulkan disabled, keeping output visible even on timeout.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from run_pr108_perplexity import select_gpu


def probe(name, command, output, env, timeout=60):
    print(f"Starting {name}: {command}", flush=True)
    started = time.monotonic()
    with (output / f"{name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"{name} timed out; capturing stacks for pid {process.pid}", flush=True)
            with (output / f"{name}.stacks.log").open("w") as stacks:
                debugger = subprocess.Popen(
                    ["sudo", "-n", "gdb", "-nx", "-batch", "-p", str(process.pid),
                     "-ex", "set pagination off", "-ex", "thread apply all bt", "-ex", "detach"],
                    stdout=stacks, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    debugger.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(debugger.pid, signal.SIGKILL)
                    debugger.wait()
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    print((output / f"{name}.log").read_text(errors="replace"), flush=True)
    result = {"name": name, "exit_code": process.returncode, "timed_out": timed_out,
              "seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-package", type=Path, required=True)
    parser.add_argument("--fixed-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"runner": os.environ.get("RUNNER_NAME"), "hostname": platform.node(),
                "gpu": select_gpu(), "platform": platform.platform()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ.copy()
    results = []
    for variant, package in (("original", args.original_package), ("fixed", args.fixed_package)):
        for mode in ("default", "no-vulkan"):
            probe_env = env.copy()
            if mode == "no-vulkan":
                probe_env["GGML_DISABLE_VULKAN"] = "1"
            results.append(probe(f"{variant}-{mode}",
                                 [str(package.resolve() / "bin/llama-perplexity"), "--list-devices"],
                                 output, probe_env))
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    results.append(probe("processes", ["ps", "-eo", "pid,ppid,stat,etimes,wchan:32,args"], output, env))
    results.append(probe("kernel", ["sudo", "-n", "dmesg", "--ctime", "--level=err,warn"], output, env))
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["timed_out"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
