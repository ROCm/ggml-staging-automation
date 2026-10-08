#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check gfx1151 capacity using Linux MemTotal plus firmware-reserved VRAM.

These are the same unprivileged counters read by Lemonade's LinuxSystemInfo
get_physical_memory() and get_amd_vram():
https://github.com/AaronStGeorge/lemonade/blob/a655fffbf750871793bf81fba0fd266ec5aa9d07/src/cpp/server/system_info.cpp
Lemonade reports them separately; their sum and the threshold are CI policy.
GTT overlaps system RAM and is excluded. This estimates physical capacity,
not the memory a particular backend can allocate.
"""

import argparse
from pathlib import Path

# Allow firmware/kernel reservations on 64 GiB machines.
FULL_MODEL_MINIMUM_GIB = 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tier", choices=("smoke", "full"), required=True)
    parser.add_argument("--gpu-target", choices=("gfx1151",), required=True)
    args = parser.parse_args()
    is_full_tier = args.model_tier == "full"

    try:
        memory = dict(
            line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
        )
        ram_gib = int(memory["MemTotal"].split()[0]) / 2**20
        vram_files = list(
            Path("/sys/bus/pci/drivers/amdgpu").glob("????:??:??.?/mem_info_vram_total")
        )
        if len(vram_files) != 1:
            raise ValueError(f"Expected one AMDGPU VRAM counter; found {len(vram_files)}")
        vram_gib = int(vram_files[0].read_text()) / 2**30
    except (OSError, KeyError, ValueError) as error:
        annotation = "error" if is_full_tier else "warning"
        print(f"::{annotation}::Cannot determine runner memory: {error}")
        return int(is_full_tier)

    total_gib = ram_gib + vram_gib
    print(
        f"{args.gpu_target}: RAM {ram_gib:.2f} + reserved VRAM {vram_gib:.2f}"
        f" = {total_gib:.2f} GiB"
    )
    is_undersized = total_gib < FULL_MODEL_MINIMUM_GIB
    reject_runner = is_full_tier and is_undersized
    if reject_runner:
        print(f"::error::Full-model benchmarks require at least {FULL_MODEL_MINIMUM_GIB} GiB.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
