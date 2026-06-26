#!/usr/bin/env python3
"""Configure, build, and install llama.cpp with the HRX backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hrx_build import (
    REPO_ROOT,
    add_common_path_args,
    cmake_build_type_arg,
    cmake_generator_args,
    cmake_toolchain_args,
    require_rocm_root,
    require_submodule,
    rocm_env,
    remove_tree,
    run,
    targets_cmake_list,
)


def build_llama_cpp(
    *,
    rocm_root: Path,
    hrx_install: Path,
    build_dir: Path,
    install_dir: Path,
    build_type: str,
    target: str,
    install: bool,
    ggml_build_tests: bool,
    llama_build_tests: bool,
    llama_tests_install: bool,
    build_examples: bool,
    backend_dl: bool,
    extra_cmake_args: list[str],
) -> None:
    llama_cpp = REPO_ROOT / "llama.cpp"
    require_submodule(llama_cpp)
    rocm_root = rocm_root.resolve()
    hrx_install = hrx_install.resolve()
    build_dir = build_dir.resolve()
    install_dir = install_dir.resolve()
    require_rocm_root(rocm_root)
    if not (hrx_install / "lib" / "cmake" / "hrx" / "hrx-config.cmake").exists():
        raise SystemExit(f"Missing hrx-system install tree: {hrx_install}")
    if not (hrx_install / "lib" / "cmake" / "loomc" / "loomc-config.cmake").exists():
        raise SystemExit(f"Missing loomc package in hrx-system install tree: {hrx_install}")

    env = rocm_env(rocm_root)
    cmake_prefix_paths = [os.fspath(hrx_install), os.fspath(rocm_root)]
    vulkan_sdk = env.get("VULKAN_SDK")
    if vulkan_sdk:
        cmake_prefix_paths.append(vulkan_sdk)
    cmake_prefix_path = ";".join(cmake_prefix_paths)
    cmake_args = [
        "cmake",
        "-S",
        llama_cpp,
        "-B",
        build_dir,
        *cmake_generator_args(),
        cmake_build_type_arg(build_type),
        f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        *cmake_toolchain_args(rocm_root),
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        r"-DCMAKE_BUILD_RPATH=$ORIGIN",
        r"-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/../lib",
        "-DGGML_CPU=ON",
        "-DGGML_VULKAN=ON",
        "-DGGML_HRX=ON",
        "-DGGML_HRX_EMBED_ROCM_LIBS=ON",
        f"-DGGML_HRX_ROCM_PATH={rocm_root}",
        f"-DGGML_HRX_AMDGPU_TARGETS={targets_cmake_list()}",
        f"-DGGML_HRX_EMBED_LIBRARY_DIRS={hrx_install / 'lib'};{rocm_root / 'lib'};{rocm_root / 'lib' / 'llvm' / 'lib'}",
        "-DGGML_HRX_BUILD_HIP_BENCHES=OFF",
        f"-DGGML_BUILD_TESTS={'ON' if ggml_build_tests else 'OFF'}",
        f"-DGGML_BUILD_EXAMPLES={'ON' if build_examples else 'OFF'}",
        f"-DLLAMA_BUILD_TESTS={'ON' if llama_build_tests else 'OFF'}",
        f"-DLLAMA_TESTS_INSTALL={'ON' if llama_tests_install else 'OFF'}",
        f"-DLLAMA_BUILD_EXAMPLES={'ON' if build_examples else 'OFF'}",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        f"-DGGML_BACKEND_DL={'ON' if backend_dl else 'OFF'}",
        *extra_cmake_args,
    ]
    run(cmake_args, env=env)
    run(["cmake", "--build", build_dir, "--target", target], env=env)
    if install:
        if install_dir.exists():
            remove_tree(install_dir)
        run(
            [
                "cmake",
                "--install",
                build_dir,
                "--prefix",
                install_dir,
            ],
            env=env,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--target", default="all")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument("--build-examples", action="store_true")
    parser.add_argument("--backend-dl", action="store_true")
    args, extra_cmake_args = parser.parse_known_args()

    build_llama_cpp(
        rocm_root=args.rocm_root,
        hrx_install=args.hrx_install_dir,
        build_dir=args.llama_build_dir,
        install_dir=args.llama_install_dir,
        build_type=args.build_type,
        target=args.target,
        install=not args.no_install,
        ggml_build_tests=args.build_tests,
        llama_build_tests=args.build_tests,
        llama_tests_install=args.build_tests,
        build_examples=args.build_examples,
        backend_dl=args.backend_dl,
        extra_cmake_args=extra_cmake_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
