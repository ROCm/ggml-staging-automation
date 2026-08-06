#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure and build the Lemonade executables used by HRX benchmarks."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from hrx_build import cmake_build_type_arg, cmake_generator_args, run


def require_lemonade_checkout(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Missing Lemonade checkout: {source_dir}")

    required = (
        source_dir / "setup.sh",
        source_dir / "CMakeLists.txt",
        source_dir / "CMakePresets.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Lemonade checkout is missing required files:\n  "
            + "\n  ".join(os.fspath(path) for path in missing)
        )

    result = subprocess.run(
        ["git", "-C", os.fspath(source_dir), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git rev-parse failed"
        raise SystemExit(f"Lemonade source is not a Git checkout: {detail}")

    checkout_root = Path(result.stdout.strip()).resolve()
    if checkout_root != source_dir:
        raise SystemExit(
            f"Lemonade source must be the Git checkout root: "
            f"expected {source_dir}, got {checkout_root}"
        )
    if not os.access(source_dir / "setup.sh", os.X_OK):
        raise SystemExit(f"Lemonade setup script is not executable: {source_dir / 'setup.sh'}")


def require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Lemonade build did not produce an executable: {path}")


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
    require_lemonade_checkout(source_dir)

    setup_env = dict(os.environ)
    setup_env["LEMONADE_SKIP_FRONTEND_DEPS"] = "1"
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

    executables = (build_dir / "lemond", build_dir / "lemonade")
    for executable in executables:
        require_executable(executable)
    print(
        "Lemonade executables:\n  "
        + "\n  ".join(os.fspath(path) for path in executables),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
