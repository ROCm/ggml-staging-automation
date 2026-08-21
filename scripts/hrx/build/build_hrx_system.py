#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure, build, and install hrx-system for llama.cpp consumption."""

from __future__ import annotations

import argparse

from hrx_build import (
    REPO_ROOT,
    add_common_path_args,
    cmake_build_type_arg,
    cmake_generator_args,
    cmake_toolchain_args,
    require_rocm_root,
    require_submodule,
    rocm_env,
    run,
    targets_cmake_list,
)


LOOM_TOOL_TARGETS = (
    "loom_tools_loom-link_loom-link",
    "loom_tools_loom-format_loom-format",
)
LOOM_TOOL_INSTALL_COMPONENTS = (
    "IREETool-loom-link",
    "IREETool-loom-format",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--target", default="all")
    parser.add_argument("--install-component", default="HrxPublicDist")
    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument("--build-benchmarks", action="store_true")
    parser.add_argument("--passthrough", action="store_true")
    args, extra_cmake_args = parser.parse_known_args()

    hrx_system = REPO_ROOT / "hrx-system"
    require_submodule(hrx_system)
    rocm_root = args.rocm_root.resolve()
    require_rocm_root(rocm_root)
    env = rocm_env(rocm_root)

    cmake_args = [
        "cmake",
        "-S",
        hrx_system,
        "-B",
        args.hrx_build_dir.resolve(),
        *cmake_generator_args(),
        cmake_build_type_arg(args.build_type),
        f"-DCMAKE_PREFIX_PATH={rocm_root}",
        f"-DCMAKE_INSTALL_PREFIX={args.hrx_install_dir.resolve()}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        *cmake_toolchain_args(rocm_root),
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        "-DLOOM_BUILD=ON",
        "-DLOOM_TARGET_AMDGPU=ON",
        "-DLOOM_TARGET_AMDGPU_TARGETS=iree_hal",
        "-DIREE_HAL_DRIVER_AMDGPU=ON",
        f"-DIREE_HAL_AMDGPU_TARGETS={targets_cmake_list()}",
        f"-DIREE_ROCM_PATH={rocm_root}",
        "-DIREE_ENABLE_LIBBACKTRACE=OFF",
        f"-DIREE_BUILD_TESTS={'ON' if args.build_tests else 'OFF'}",
        f"-DIREE_BUILD_BENCHMARKS={'ON' if args.build_benchmarks else 'OFF'}",
        "-DLIBHRX_BUILD=ON",
        f"-DLIBHRX_BUILD_CTS={'ON' if args.build_tests else 'OFF'}",
        f"-DHRX_INSTALL_TESTS={'ON' if args.build_tests else 'OFF'}",
        f"-DLIBHRX_BUILD_PASSTHROUGH={'ON' if args.passthrough else 'OFF'}",
        *extra_cmake_args,
    ]
    run(cmake_args, env=env)
    run(
        [
            "cmake",
            "--build",
            args.hrx_build_dir.resolve(),
            "--target",
            args.target,
            *LOOM_TOOL_TARGETS,
        ],
        env=env,
    )
    install_components = dict.fromkeys(
        (args.install_component, *LOOM_TOOL_INSTALL_COMPONENTS)
    )
    for component in install_components:
        run(
            [
                "cmake",
                "--install",
                args.hrx_build_dir.resolve(),
                "--prefix",
                args.hrx_install_dir.resolve(),
                "--component",
                component,
            ],
            env=env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
