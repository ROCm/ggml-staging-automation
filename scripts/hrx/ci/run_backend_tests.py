#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run packaged HRX correctness tests and reject successful all-skipped runs.

KFD GPU nodes establish the ROCr ordinal before any HRX executable starts.
The pinned process must then enumerate exactly one HRX device on gfx1151.
Console output is retained even on crashes; only completed supported cases
count as execution evidence, since upstream summaries also count skips.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def gpu_inventory(topology: Path) -> list[dict]:
    devices = []
    for node in sorted(topology.iterdir(), key=lambda path: int(path.name)):
        properties = dict(line.split(maxsplit=1) for line in (node / "properties").read_text().splitlines())
        target = int(properties.get("gfx_target_version", "0"))
        if target == 0:
            continue
        devices.append({
            "ordinal": len(devices),
            "node": node.name,
            "name": (node / "name").read_text().strip(),
            "gpu_id": (node / "gpu_id").read_text().strip(),
            "properties": properties,
        })
    return devices


def hrx_device(listing: str) -> str:
    devices = re.findall(r"^\s*(HRX\d+):\s*(.+)$", listing, re.MULTILINE)
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one visible HRX GPU, got {devices}")
    name, description = devices[0]
    if "(gfx1151)" not in description:
        raise RuntimeError(f"Pinned HRX device is not gfx1151: {description}")
    return name


def test_evidence(log: str, device: str) -> dict:
    text = re.sub(r"\x1b\[[0-9;]*m", "", log)
    sections = re.split(r"^Backend \d+/\d+: ", text, flags=re.MULTILINE)[1:]
    selected = [section for section in sections if section.splitlines()[0] == device]
    if len(selected) != 1:
        raise RuntimeError(f"Missing unique test section for {device}")
    section = selected[0]
    if "  Device description:" not in section:
        raise RuntimeError(f"{device} was skipped or did not initialize")
    cases = re.findall(r"^  (\S+\(.*\)): (OK|FAIL)$", section, re.MULTILINE)
    passed = sum(status == "OK" for _, status in cases)
    failures = [case for case, status in cases if status == "FAIL"]
    summaries = re.findall(r"^  (\d+)/(\d+) tests passed$", section, re.MULTILINE)
    status = re.search(rf"^  Backend {re.escape(device)}: (OK|FAIL)$", section, re.MULTILINE)
    completed = bool(summaries) and status is not None
    return {
        "executed": len(cases),
        "passed": passed,
        "failed": len(failures),
        "failing_cases": failures,
        "unsupported": len(re.findall(r"^  .*: not supported ", section, re.MULTILINE)),
        "suite_completed": completed,
        "backend_status": status[1] if status else None,
        "upstream_summary": summaries[-1] if summaries else None,
    }


def capture(command: list[str], destination: Path, env: dict[str, str], timeout: int) -> int:
    print("++", " ".join(command), flush=True)
    with destination.open("w") as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = args.package_dir.resolve()
    metadata = {"outcome": "infrastructure_failure", "commands": []}
    result = 1
    try:
        for name, revision in (("automation", "HEAD"), ("llama_cpp", "HEAD:llama.cpp"), ("hrx_system", "HEAD:hrx-system")):
            metadata[name] = subprocess.check_output(["git", "rev-parse", revision], text=True).strip()
        inventory = gpu_inventory(Path("/sys/class/kfd/kfd/topology/nodes"))
        metadata["gpu_inventory"] = inventory
        supported = [gpu for gpu in inventory if gpu["properties"]["gfx_target_version"] == "110501"]
        if not supported:
            raise RuntimeError("No supported gfx1151 GPU in KFD topology")
        gpu = supported[0]
        metadata["selected_gpu"] = gpu
        env = dict(os.environ)
        env["ROCR_VISIBLE_DEVICES"] = str(gpu["ordinal"])
        metadata["ROCR_VISIBLE_DEVICES"] = env["ROCR_VISIBLE_DEVICES"]
        command = [str(package / "bin/llama-server"), "--list-devices"]
        metadata["commands"].append(command)
        code = capture(command, output / "devices.log", env, 120)
        if code != 0:
            raise RuntimeError(f"HRX device discovery exited {code}")
        device = hrx_device((output / "devices.log").read_text())
        metadata["device"] = device
        command = ["stdbuf", "-oL", "-eL", str(package / "bin/test-backend-ops"), "test", "-b", device]
        metadata["commands"].append(command)
        # Persist identity and command evidence before a potentially long suite.
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        timed_out = False
        try:
            code = capture(command, output / "test-backend-ops.log", env, 18000)
        except subprocess.TimeoutExpired:
            code = None
            timed_out = True
        metadata["returncode"] = code
        metadata["timed_out"] = timed_out
        evidence = test_evidence((output / "test-backend-ops.log").read_text(errors="replace"), device)
        metadata.update(evidence)
        if evidence["executed"] == 0:
            raise RuntimeError("No completed supported HRX correctness cases; refusing an all-skipped result")
        suite_completed = evidence["suite_completed"]
        no_failures = evidence["failed"] == 0
        backend_ok = evidence["backend_status"] == "OK"
        process_ok = code == 0
        passed = suite_completed and no_failures and backend_ok and process_ok
        metadata["outcome"] = "passed" if passed else "backend_failure"
        result = 0 if passed else 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        metadata["error"] = str(error)
        print(f"Backend test infrastructure failure: {error}", flush=True)
    finally:
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary = (
            f"HRX backend correctness: **{metadata['outcome']}**\n\n"
            f"Device: `{metadata.get('device')}`; ROCR_VISIBLE_DEVICES: `{metadata.get('ROCR_VISIBLE_DEVICES')}`\n\n"
            f"Executed: {metadata.get('executed', 0)}; passed: {metadata.get('passed', 0)}; "
            f"failed: {metadata.get('failed', 0)}; suite completed: {metadata.get('suite_completed', False)}\n\n"
            "See the hrx-backend-tests-gfx1151 artifact for revisions, GPU identity, commands, and full logs.\n"
        )
        (output / "summary.md").write_text(summary)
        print(summary, flush=True)
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as step_summary:
                step_summary.write(summary)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
