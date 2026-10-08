#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Report Linux memory and reject undersized full-model benchmark runners.

On APUs, GTT uses system RAM and GPU-reported pools can overlap. Keep the
kernel counters separate: their sum is not a reliable physical-memory total
or a guarantee that an HRX allocation will succeed. Use firmware-reported
installed RAM for the capacity gate, before firmware's CPU/GPU reservation.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
from pathlib import Path

FULL_MODEL_MINIMUM_GIB = 64


def format_memory(size_bytes: int) -> str:
    return f"{size_bytes / 2**30:.2f} GiB ({size_bytes} bytes)"


def report_counter(path: Path, *, unit_bytes: int = 1) -> None:
    try:
        value = int(path.read_text()) * unit_bytes
    except (OSError, ValueError) as error:
        # Optional driver counters vary by kernel and runner permissions.
        print(f"{path}: unavailable ({error})")
        return
    print(f"{path}: {format_memory(value)}")


def installed_memory() -> int:
    result = subprocess.run(
        ["sudo", "-n", "dmidecode", "--type", "17"],
        check=True,
        capture_output=True,
        text=True,
    )
    sizes = re.findall(r"^\s*Size: (.+)$", result.stdout, re.MULTILINE)
    if not sizes:
        raise ValueError("SMBIOS has no memory-device sizes")
    total = 0
    units = {"kB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    for size in sizes:
        if size == "No Module Installed":
            continue
        match = re.fullmatch(r"(\d+) (kB|MB|GB|TB)", size)
        if match is None:
            raise ValueError(f"Unknown SMBIOS memory-device size: {size}")
        total += int(match[1]) * units[match[2]]
    if total == 0:
        raise ValueError("SMBIOS reports no installed memory")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tier", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    print(f"Runner: {os.environ.get('RUNNER_NAME', platform.node())}")
    print(f"Host: {platform.node()}; kernel: {platform.release()}")
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            print(f"CPU: {line.split(':', 1)[1].strip()}")
            break

    print("\nLinux memory (excludes firmware-reserved memory):")
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            print(f"{name}: {format_memory(int(value.split()[0]) * 1024)}")

    print("\nAMDGPU memory counters (GTT overlaps system RAM; do not sum pools):")
    devices = sorted(Path("/sys/bus/pci/drivers/amdgpu").glob("????:??:??.?"))
    if not devices:
        print("No AMDGPU PCI devices exposed in sysfs.")
    for device in devices:
        for counter in ("vram_total", "vram_used", "gtt_total", "gtt_used"):
            report_counter(device / f"mem_info_{counter}")

    print("\nTTM shared-memory limit (a budget, not additional RAM):")
    report_counter(
        Path("/sys/module/ttm/parameters/pages_limit"),
        unit_bytes=os.sysconf("SC_PAGE_SIZE"),
    )

    try:
        capacity = installed_memory()
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"\nInstalled RAM: unavailable ({error})")
        if args.model_tier == "full":
            print("::error::Cannot verify installed RAM for full-model benchmarks.")
            return 1
        return 0

    print(f"\nInstalled RAM (SMBIOS): {format_memory(capacity)}")
    is_full_tier = args.model_tier == "full"
    is_undersized = capacity < FULL_MODEL_MINIMUM_GIB * 2**30
    reject_runner = is_full_tier and is_undersized
    if reject_runner:
        print(
            f"::error::Full-model benchmarks require at least "
            f"{FULL_MODEL_MINIMUM_GIB} GiB installed RAM; "
            f"this runner has {format_memory(capacity)}."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
