#!/usr/bin/env python3
"""Run the pinned ROCm fetch plus HRX and llama.cpp build/install steps."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrx_build import REPO_ROOT, add_common_path_args, python_executable, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--build-type", default="Release", help="llama.cpp build type")
    parser.add_argument("--hrx-build-type", default="Release", help="hrx-system build type")
    parser.add_argument(
        "--vulkan-sdk-dir",
        type=Path,
        default=None,
        help=(
            "Vulkan SDK install root with include, lib, and bin subdirectories; "
            "omit to build llama.cpp without Vulkan"
        ),
    )
    parser.add_argument(
        "--build-test-package",
        action="store_true",
        help="also install llama.cpp test binaries into the llama.cpp install tree",
    )
    args, extra_args = parser.parse_known_args()

    common = [
        "--rocm-root",
        args.rocm_root.resolve(),
        "--download-cache-dir",
        args.download_cache_dir.resolve(),
        "--hrx-build-dir",
        args.hrx_build_dir.resolve(),
        "--hrx-install-dir",
        args.hrx_install_dir.resolve(),
        "--llama-build-dir",
        args.llama_build_dir.resolve(),
        "--llama-install-dir",
        args.llama_install_dir.resolve(),
        "--package-dir",
        args.package_dir.resolve(),
    ]
    vulkan_sdk_args = []
    if args.vulkan_sdk_dir is not None:
        vulkan_sdk_args = ["--vulkan-sdk-dir", args.vulkan_sdk_dir.resolve()]
    if not args.skip_fetch:
        run([python_executable(), REPO_ROOT / "scripts" / "hrx" / "fetch_rocm.py", *common])
    run(
        [
            python_executable(),
            REPO_ROOT / "scripts" / "hrx" / "build_hrx_system.py",
            *common,
            "--build-type",
            args.hrx_build_type,
            *extra_args,
        ]
    )
    run(
        [
            python_executable(),
            REPO_ROOT / "scripts" / "hrx" / "build_llama_cpp.py",
            *common,
            "--build-type",
            args.build_type,
            *vulkan_sdk_args,
            *(["--build-installed-tests"] if args.build_test_package else []),
        ]
    )
    if not args.skip_validate:
        validate_args = ["--require-vulkan-loader"] if args.vulkan_sdk_dir is not None else []
        run(
            [
                python_executable(),
                REPO_ROOT / "scripts" / "hrx" / "validate_install.py",
                *common,
                *validate_args,
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
