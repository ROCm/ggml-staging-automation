#!/usr/bin/env python3
"""Capture GPU access and package loading on canary runners without models."""

import argparse
import glob
import json
import os
from pathlib import Path
import subprocess


def capture(name, command, output, env):
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        text = result.stdout + result.stderr
        code = result.returncode
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode(errors="replace")
        text += (exc.stderr or b"").decode(errors="replace")
        code = "timeout"
    (output / f"{name}.log").write_text(text)
    print(f"{name}: {code}\n{text}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    keys = ("RUNNER_NAME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "GGML_DISABLE_VULKAN",
            "GGML_VK_VISIBLE_DEVICES", "VK_ICD_FILENAMES", "VK_DRIVER_FILES",
            "LD_LIBRARY_PATH", "LD_PRELOAD")
    metadata = {key: os.environ.get(key) for key in keys}
    metadata["uid"] = os.getuid()
    metadata["groups"] = os.getgroups()
    devices = sorted(glob.glob("/dev/dri/*") + glob.glob("/dev/kfd"))
    metadata["devices"] = {}
    for path in devices:
        info = os.stat(path)
        metadata["devices"][path] = {
            "mode": oct(info.st_mode), "uid": info.st_uid, "gid": info.st_gid,
            "read_write": os.access(path, os.R_OK | os.W_OK),
        }
    metadata["topology"] = {
        path: Path(path).read_text()
        for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    package = args.package.resolve()
    binary = str(package / "bin/llama-perplexity")
    env = dict(os.environ)
    env["GGML_HRX_LOG_DEVICE_EVENTS"] = "1"
    capture("devices-inherited", [binary, "--list-devices"], args.output, env)
    # Compare the same UID with the device's owning group, without changing
    # device permissions or the runner account's persistent group membership.
    if Path("/dev/kfd").exists():
        device_group = os.stat("/dev/kfd").st_gid
        capture("devices-with-gpu-group",
                ["sudo", "-n", "setpriv", "--reuid", str(os.getuid()),
                 "--regid", str(os.getgid()), "--groups",
                 ",".join(str(group) for group in sorted(set(os.getgroups() + [device_group]))),
                 binary, "--list-devices"], args.output, env)
    capture("dependencies", ["ldd", binary], args.output, env)
    env["VK_LOADER_DEBUG"] = "error,warn"
    env["LD_DEBUG"] = "libs"
    capture("loader", [binary, "--list-devices"], args.output, env)


if __name__ == "__main__":
    main()
