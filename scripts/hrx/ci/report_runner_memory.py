#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Report Linux memory and reject undersized full-model benchmark runners.

CI specifies the runner's memory layout and minimum full-model capacity.
Unified-memory runners add Linux MemTotal and firmware-reserved VRAM;
discrete GPUs use VRAM alone. GTT overlaps system RAM and is not added.
All counters are read without elevated privileges. Passing this capacity
check does not guarantee that every backend allocation will succeed.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tier", choices=("smoke", "full"), required=True)
    parser.add_argument("--memory-layout", choices=("unified", "discrete"), required=True)
    parser.add_argument("--minimum-gib", type=int, required=True)
    args = parser.parse_args()
    print(f"Runner: {os.environ.get('RUNNER_NAME', platform.node())}")
    print(f"Host: {platform.node()}; kernel: {platform.release()}")
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            print(f"CPU: {line.split(':', 1)[1].strip()}")
            break

    print("\nLinux memory (excludes firmware-reserved memory):")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[name] = int(value.split()[0]) * 1024
            print(f"{name}: {format_memory(memory[name])}")

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
        if len(devices) != 1:
            raise ValueError(f"Expected one AMDGPU device; found {len(devices)}")
        capacity = int((devices[0] / "mem_info_vram_total").read_text())
        capacity_label = "Dedicated VRAM"
        if args.memory_layout == "unified":
            capacity += memory["MemTotal"]
            capacity_label = "MemTotal + physical APU VRAM"
    except (OSError, KeyError, ValueError) as error:
        print(f"\nMemory capacity: unavailable ({error})")
        if args.model_tier == "full":
            print("::error::Cannot verify memory capacity for full-model benchmarks.")
            return 1
        return 0

    print(f"\n{capacity_label}: {format_memory(capacity)}")
    is_full_tier = args.model_tier == "full"
    is_undersized = capacity < args.minimum_gib * 2**30
    reject_runner = is_full_tier and is_undersized
    if reject_runner:
        print(
            f"::error::Full-model benchmarks require at least "
            f"{args.minimum_gib} GiB of {capacity_label}; "
            f"this runner has {format_memory(capacity)}."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
