#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure, build, and install llama.cpp with the HRX backend."""

from __future__ import annotations

import argparse
import json
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
)


def vulkan_cmake_args(vulkan_sdk_dir: Path) -> list[str]:
    include_dir = vulkan_sdk_dir / "include"
    library = vulkan_sdk_dir / "lib" / "libvulkan.so"
    glslc = vulkan_sdk_dir / "bin" / "glslc"
    spirv_headers_dir = vulkan_sdk_dir / "share" / "cmake" / "SPIRV-Headers"
    required = [
        include_dir / "vulkan" / "vulkan.h",
        include_dir / "spirv" / "unified1" / "spirv.hpp",
        library,
        glslc,
        spirv_headers_dir / "SPIRV-HeadersConfig.cmake",
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
        f"-DSPIRV-Headers_DIR={spirv_headers_dir}",
    ]


VULKAN_RUNTIME_GLOBS = (
    # Vulkan loader
    "libvulkan.so*",
    # RADV driver and its libdrm dependencies; the loader discovers the driver
    # through the bundled ICD manifest (VK_DRIVER_FILES).
    "libvulkan_radeon.so",
    "libdrm.so*",
    "libdrm_amdgpu.so*",
)

VULKAN_ICD_MANIFEST = Path("share") / "vulkan" / "icd.d" / "radeon_icd.x86_64.json"


def copy_vulkan_runtime(vulkan_sdk_dir: Path, dest_dir: Path) -> None:
    lib_dir = vulkan_sdk_dir / "lib"
    for required in ("libvulkan.so.1", "libvulkan_radeon.so", "libdrm_amdgpu.so.1"):
        if not (lib_dir / required).exists():
            raise SystemExit(f"Missing Vulkan runtime library: {lib_dir / required}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for pattern in VULKAN_RUNTIME_GLOBS:
        for source in sorted(lib_dir.glob(pattern)):
            dest = dest_dir / source.name
            if dest.exists() or dest.is_symlink():
                remove_tree(dest)
            shutil.copy2(source, dest, follow_symlinks=False)


def write_vulkan_icd_manifest(vulkan_sdk_dir: Path, install_dir: Path) -> None:
    """Write an ICD manifest whose library_path resolves inside the install tree."""
    sdk_manifest = vulkan_sdk_dir / VULKAN_ICD_MANIFEST
    if not sdk_manifest.exists():
        raise SystemExit(f"Missing Vulkan ICD manifest: {sdk_manifest}")
    manifest = json.loads(sdk_manifest.read_text(encoding="utf-8"))
    # Relative library paths are resolved against the manifest's directory.
    manifest["ICD"]["library_path"] = "../../../lib/libvulkan_radeon.so"
    dest = install_dir / VULKAN_ICD_MANIFEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")


def remove_vulkan_artifacts(dest_dir: Path) -> None:
    if not dest_dir.exists():
        return
    for pattern in (*VULKAN_RUNTIME_GLOBS, "libggml-vulkan.so*"):
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
        # Don't assume CPU instructions available on the build machine are
        # available on the host running this build.
        "-DGGML_NATIVE=OFF",
        f"-DGGML_VULKAN={'ON' if vulkan_sdk_dir is not None else 'OFF'}",
        "-DGGML_HRX=ON",
        "-DGGML_HRX_BUNDLE_RUNTIME_LIBS=ON",
        f"-DGGML_HRX_BUNDLE_LIBRARY_DIRS={hrx_install / 'lib'};{rocm_root / 'lib'};{rocm_root / 'lib' / 'llvm' / 'lib'}",
        f"-DGGML_BUILD_TESTS={'ON' if ggml_build_tests else 'OFF'}",
        f"-DGGML_BUILD_EXAMPLES={'ON' if build_examples else 'OFF'}",
        f"-DLLAMA_BUILD_TESTS={'ON' if llama_build_tests else 'OFF'}",
        f"-DLLAMA_TESTS_INSTALL={'ON' if llama_build_tests else 'OFF'}",
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
        copy_vulkan_runtime(vulkan_sdk_dir, build_dir / "bin")
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
            copy_vulkan_runtime(vulkan_sdk_dir, install_dir / "lib")
            write_vulkan_icd_manifest(vulkan_sdk_dir, install_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--target", default="all")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument(
        "--build-installed-tests",
        action="store_true",
        help="build and install llama.cpp tests",
    )
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
        llama_build_tests=args.build_tests or args.build_installed_tests,
        build_examples=args.build_examples,
        backend_dl=args.backend_dl,
        extra_cmake_args=extra_cmake_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
