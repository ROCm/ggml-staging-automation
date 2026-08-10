#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure and build the Lemonade executables used by HRX benchmarks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hrx_build import cmake_build_type_arg, cmake_generator_args, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Lemonade Git checkout root",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="CMake build directory",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()

    setup_env = dict(os.environ)
    setup_env["LEMONADE_SKIP_FRONTEND_DEPS"] = "1"
    # setup.sh only installs missing dependencies non-interactively when it
    # detects CI; the variable is not always propagated into containerized
    # runner steps.
    setup_env["CI"] = "1"
    run(["./setup.sh"], cwd=source_dir, env=setup_env)

    run(
        [
            "cmake",
            "-S",
            source_dir,
            "-B",
            build_dir,
            *cmake_generator_args(),
            cmake_build_type_arg("Release"),
            "-DBUILD_WEB_APP=OFF",
        ],
        cwd=source_dir,
    )
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--target",
            "lemond",
            "lemonade",
            "--parallel",
            "4",
        ],
        cwd=source_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
