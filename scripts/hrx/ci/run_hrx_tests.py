#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the packaged CTest manifest on a pinned gfx1151 and retain evidence."""

import json
import os
from pathlib import Path
import subprocess
import time


def select_gpu():
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    ordinal = 0
    for node in sorted(nodes.iterdir(), key=lambda path: int(path.name)):
        properties = dict(line.split(maxsplit=1) for line in (node / "properties").read_text().splitlines())
        target = int(properties.get("gfx_target_version", "0"))
        if target == 0:
            continue
        if target == 110501:
            return str(ordinal), properties
        ordinal += 1
    raise SystemExit("No gfx1151 GPU found in KFD topology")


def main():
    root = Path("release/hrx-tests").resolve()
    results = root / "results"
    results.mkdir(exist_ok=True)
    ordinal, properties = select_gpu()
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = ordinal
    env["LD_LIBRARY_PATH"] = str(root / "lib")
    command = ["ctest", "--test-dir", str(root), "--parallel", "2", "--verbose",
               "--output-junit", str(results / "junit.xml"), "--timeout", "900"]
    metadata = json.loads((root / "metadata.json").read_text())
    metadata.update({"gpu_properties": properties, "ROCR_VISIBLE_DEVICES": ordinal,
                     "ctest_parallel": 2, "command": command,
                     "runtime_library_path": env["LD_LIBRARY_PATH"],
                     "runtime_cache_state": "unknown (self-hosted runner)",
                     "ctest_version": subprocess.check_output(["ctest", "--version"], text=True)})
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Resolve every packaged test before timing so loader failures are distinct.
    with (results / "dependencies.log").open("w") as log:
        for binary in sorted((root / "bin").iterdir()):
            proc = subprocess.run(["ldd", str(binary)], env=env, text=True, capture_output=True)
            log.write(f"{binary.name}\n{proc.stdout}{proc.stderr}\n")
            missing_library = "not found" in proc.stdout
            loader_failed = proc.returncode != 0
            invalid_dependencies = missing_library or loader_failed
            if invalid_dependencies:
                raise SystemExit(f"Unresolved dependencies for {binary.name}; see dependencies.log")
    start = time.monotonic()
    with (results / "ctest.log").open("w") as log:
        proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata["test_wall_seconds"] = time.monotonic() - start
    metadata["ctest_exit_code"] = proc.returncode
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print((results / "ctest.log").read_text())
    print(json.dumps(metadata, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
