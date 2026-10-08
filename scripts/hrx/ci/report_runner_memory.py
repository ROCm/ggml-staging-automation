#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Report Linux memory and reject undersized full-model benchmark runners.

CI supplies the GPU target; currently only gfx1151 is supported. Its unified
memory capacity is Linux MemTotal plus firmware-reserved VRAM. GTT overlaps
system RAM and is not added. All counters are read without elevated
privileges. Passing this capacity check does not guarantee that every
backend allocation will succeed.

Counter reads follow Lemonade's LinuxSystemInfo::get_physical_memory()
(MemTotal) and get_amd_vram() (mem_info_vram_total), without sudo:
https://github.com/AaronStGeorge/lemonade/blob/a655fffbf750871793bf81fba0fd266ec5aa9d07/src/cpp/server/system_info.cpp
Lemonade reports RAM and VRAM separately. Their sum and the minimum below
are CI policy; Lemonade's APU GPU-budget calculation uses VRAM plus GTT.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

# Allow firmware/kernel reservations on 64 GiB gfx1151 machines.
FULL_MODEL_MINIMUM_GIB = 60


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
    parser.add_argument("--gpu-target", choices=("gfx1151",), required=True)
    args = parser.parse_args()
    print(f"Runner: {os.environ.get('RUNNER_NAME', platform.node())}")
    print(f"Host: {platform.node()}; kernel: {platform.release()}")
    print(f"GPU target: {args.gpu_target}")
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
        vram = int((devices[0] / "mem_info_vram_total").read_text())
        capacity = memory["MemTotal"] + vram
    except (OSError, KeyError, ValueError) as error:
        print(f"\nMemory capacity: unavailable ({error})")
        if args.model_tier == "full":
            print("::error::Cannot verify memory capacity for full-model benchmarks.")
            return 1
        return 0

    print(f"\nMemTotal + physical APU VRAM: {format_memory(capacity)}")
    is_full_tier = args.model_tier == "full"
    is_undersized = capacity < FULL_MODEL_MINIMUM_GIB * 2**30
    reject_runner = is_full_tier and is_undersized
    if reject_runner:
        print(
            f"::error::Full-model benchmarks require at least "
            f"{FULL_MODEL_MINIMUM_GIB} GiB of MemTotal + physical APU VRAM; "
            f"this runner has {format_memory(capacity)}."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
