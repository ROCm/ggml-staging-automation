#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Report Linux memory and reject undersized full-model benchmark runners.

On Strix Halo, Linux MemTotal excludes firmware-reserved VRAM. Add only the
APU's physical VRAM counter to estimate capacity; GTT overlaps system RAM.
This filters undersized machines, but does not guarantee an HRX allocation
will succeed. All counters are read without elevated privileges.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

# Accept 64 GiB machines with some memory reserved for firmware and the kernel.
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


def strix_halo_vram() -> int:
    devices = []
    for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        properties = dict(line.split() for line in path.read_text().splitlines())
        if properties.get("gfx_target_version") != "110501":
            continue
        domain = int(properties["domain"])
        location = int(properties["location_id"])
        bus = location >> 8
        slot = (location & 0xFF) >> 3
        function = location & 7
        devices.append(f"{domain:04x}:{bus:02x}:{slot:02x}.{function}")
    if len(devices) != 1:
        raise ValueError(f"Expected one Strix Halo APU; found {devices}")
    device = Path("/sys/bus/pci/devices") / devices[0]
    print(f"Strix Halo PCI device: {devices[0]}")
    return int((device / "mem_info_vram_total").read_text())


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
        capacity = memory["MemTotal"] + strix_halo_vram()
    except (OSError, KeyError, ValueError) as error:
        print(f"\nStrix Halo memory capacity: unavailable ({error})")
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
