#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Execute an HRX CTest package using the runner's environment."""

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()
    root = args.package_dir.resolve()
    return subprocess.call([
        "ctest", "--test-dir", str(root), "--parallel", "2", "--verbose",
        "--output-junit", str(root / "junit.xml"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
