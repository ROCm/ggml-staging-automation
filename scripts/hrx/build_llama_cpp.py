#!/usr/bin/env python3
"""Configure, build, and install llama.cpp with the HRX backend."""

from __future__ import annotations

import argparse
import os
import shutil
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


def vulkan_cmake_args(vulkan_sdk_dir: Path) -> list[str]:
    include_dir = vulkan_sdk_dir / "include"
    library = vulkan_sdk_dir / "lib" / "libvulkan.so"
    glslc = vulkan_sdk_dir / "bin" / "glslc"
    required = [
        include_dir / "vulkan" / "vulkan.h",
        library,
        glslc,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Vulkan SDK is incomplete. Missing:\n  "
            + "\n  ".join(os.fspath(path) for path in missing)
        )
    return [
        f"-DVulkan_INCLUDE_DIR={include_dir}",
        f"-DVulkan_LIBRARY={library}",
        f"-DVulkan_GLSLC_EXECUTABLE={glslc}",
    ]


def copy_vulkan_loader(vulkan_sdk_dir: Path, dest_dir: Path) -> None:
    lib_dir = vulkan_sdk_dir / "lib"
    required_loader = lib_dir / "libvulkan.so.1"
    if not required_loader.exists():
        raise SystemExit(f"Missing Vulkan loader: {required_loader}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(lib_dir.glob("libvulkan.so*")):
        dest = dest_dir / source.name
        if dest.exists() or dest.is_symlink():
            remove_tree(dest)
        shutil.copy2(source, dest, follow_symlinks=False)


def remove_vulkan_artifacts(dest_dir: Path) -> None:
    if not dest_dir.exists():
        return
    for pattern in ("libvulkan.so*", "libggml-vulkan.so*"):
        for dest in dest_dir.glob(pattern):
            remove_tree(dest)


def build_llama_cpp(
    *,
    rocm_root: Path,
    hrx_install: Path,
    vulkan_sdk_dir: Path | None,
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
    vulkan_args: list[str] = []
    if vulkan_sdk_dir is not None:
        vulkan_sdk_dir = vulkan_sdk_dir.resolve()
        vulkan_args = vulkan_cmake_args(vulkan_sdk_dir)
    require_rocm_root(rocm_root)
    if not (hrx_install / "lib" / "cmake" / "hrx" / "hrx-config.cmake").exists():
        raise SystemExit(f"Missing hrx-system install tree: {hrx_install}")
    if not (hrx_install / "lib" / "cmake" / "loomc" / "loomc-config.cmake").exists():
        raise SystemExit(f"Missing loomc package in hrx-system install tree: {hrx_install}")

    env = rocm_env(rocm_root)
    cmake_prefix_path = ";".join([os.fspath(hrx_install), os.fspath(rocm_root)])
    cmake_args = [
        "cmake",
        "-S",
        llama_cpp,
        "-B",
        build_dir,
        *cmake_generator_args(),
        cmake_build_type_arg(build_type),
        f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
        *vulkan_args,
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        *cmake_toolchain_args(rocm_root),
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        r"-DCMAKE_BUILD_RPATH=$ORIGIN",
        r"-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/../lib",
        "-DGGML_CPU=ON",
        f"-DGGML_VULKAN={'ON' if vulkan_sdk_dir is not None else 'OFF'}",
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
    if vulkan_sdk_dir is not None:
        # llama.cpp globally configures cmake to put .so files under bin in the
        # build directory. During install it uses a more standard lib layout.
        copy_vulkan_loader(vulkan_sdk_dir, build_dir / "bin")
    else:
        remove_vulkan_artifacts(build_dir / "bin")
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
        if vulkan_sdk_dir is not None:
            copy_vulkan_loader(vulkan_sdk_dir, install_dir / "lib")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--target", default="all")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument("--build-examples", action="store_true")
    parser.add_argument("--backend-dl", action="store_true")
    parser.add_argument(
        "--vulkan-sdk-dir",
        type=Path,
        default=None,
        help=(
            "Vulkan SDK install root with include, lib, and bin subdirectories; "
            "omit to build llama.cpp without Vulkan"
        ),
    )
    args, extra_cmake_args = parser.parse_known_args()

    build_llama_cpp(
        rocm_root=args.rocm_root,
        hrx_install=args.hrx_install_dir,
        vulkan_sdk_dir=args.vulkan_sdk_dir,
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
