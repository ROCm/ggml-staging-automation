#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fetch and build the Vulkan build dependencies needed by llama.cpp."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"

VULKAN_HEADERS_REPO = "https://github.com/KhronosGroup/Vulkan-Headers.git"
VULKAN_HEADERS_TAG = "v1.3.296"
VULKAN_LOADER_REPO = "https://github.com/KhronosGroup/Vulkan-Loader.git"
VULKAN_LOADER_TAG = "v1.3.296"
SHADERC_REPO = "https://github.com/google/shaderc.git"
SHADERC_TAG = "v2024.3"
LIBDRM_REPO = "https://gitlab.freedesktop.org/mesa/drm.git"
LIBDRM_TAG = "libdrm-2.4.124"
MESA_REPO = "https://gitlab.freedesktop.org/mesa/mesa.git"
MESA_TAG = "mesa-25.1.5"


def run(
    args: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [os.fspath(arg) for arg in args]
    print("++", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clone_or_update(repo: str, tag: str, dest: Path) -> None:
    if (dest / ".git").exists():
        run(["git", "fetch", "--tags", "--force", "origin", tag], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", tag, repo, dest])
    run(["git", "checkout", "--detach", tag], cwd=dest)


def cmake_generator_args() -> list[str]:
    return ["-GNinja"] if shutil.which("ninja") else []


def cmake_configure(
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
    *extra_args: str,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "cmake",
            "-S",
            source_dir,
            "-B",
            build_dir,
            *cmake_generator_args(),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            *extra_args,
        ]
    )


def build_vulkan_headers(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    cmake_configure(source_dir, build_dir, install_dir)
    run(["cmake", "--build", build_dir, "--target", "install"])


def build_vulkan_loader(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    cmake_configure(
        source_dir,
        build_dir,
        install_dir,
        f"-DCMAKE_PREFIX_PATH={install_dir}",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_WSI_XCB_SUPPORT=OFF",
        "-DBUILD_WSI_XLIB_SUPPORT=OFF",
        "-DBUILD_WSI_WAYLAND_SUPPORT=OFF",
        "-DBUILD_WSI_DIRECTFB_SUPPORT=OFF",
    )
    run(["cmake", "--build", build_dir, "--target", "install"])


def sync_shaderc_deps(source_dir: Path) -> None:
    run(["python3", "utils/git-sync-deps"], cwd=source_dir)


def build_spirv_headers(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    cmake_configure(
        source_dir,
        build_dir,
        install_dir,
        "-DSPIRV_HEADERS_ENABLE_TESTS=OFF",
        "-DSPIRV_HEADERS_ENABLE_INSTALL=ON",
    )
    run(["cmake", "--build", build_dir, "--target", "install"])


def build_glslang_standalone(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    """Build the glslangValidator binary Mesa needs to compile driver shaders."""
    cmake_configure(
        source_dir,
        build_dir,
        install_dir,
        "-DENABLE_OPT=OFF",
        "-DENABLE_HLSL=OFF",
        "-DBUILD_EXTERNAL=OFF",
        "-DGLSLANG_TESTS=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
    )
    run(["cmake", "--build", build_dir, "--target", "glslang-standalone"])
    glslang = build_dir / "StandAlone" / "glslang"
    if not glslang.exists():
        raise SystemExit(f"glslang build did not produce standalone binary: {glslang}")
    bin_dir = install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(glslang, bin_dir / "glslangValidator")


def meson_executable() -> str:
    meson = shutil.which("meson")
    if meson is None:
        raise SystemExit("meson is required to build libdrm and Mesa RADV")
    return meson


def meson_install(
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
    *extra_args: str,
) -> None:
    meson = meson_executable()
    env = dict(os.environ)
    env["PKG_CONFIG_PATH"] = os.pathsep.join(
        [
            os.fspath(install_dir / "lib" / "pkgconfig"),
            env.get("PKG_CONFIG_PATH", ""),
        ]
    ).rstrip(os.pathsep)
    env["PATH"] = os.pathsep.join([os.fspath(install_dir / "bin"), env["PATH"]])
    reconfigure = ["--reconfigure"] if (build_dir / "meson-private").exists() else []
    run(
        [
            meson,
            "setup",
            *reconfigure,
            build_dir,
            source_dir,
            "--prefix",
            install_dir,
            "--libdir",
            "lib",
            "--buildtype",
            "release",
            # Resolve bundled dependencies (libdrm) from the same directory
            # when these libraries are shipped inside the llama.cpp install
            # tree.
            "-Dc_link_args=-Wl,-rpath,$ORIGIN",
            *extra_args,
        ],
        env=env,
    )
    run([meson, "install", "-C", build_dir], env=env)


def build_libdrm(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    meson_install(
        source_dir,
        build_dir,
        install_dir,
        "-Damdgpu=enabled",
        "-Dradeon=disabled",
        "-Dintel=disabled",
        "-Dnouveau=disabled",
        "-Dvmwgfx=disabled",
        "-Dfreedreno=disabled",
        "-Dvc4=disabled",
        "-Detnaviv=disabled",
        "-Dexynos=disabled",
        "-Domap=disabled",
        "-Dtegra=disabled",
        "-Dcairo-tests=disabled",
        "-Dman-pages=disabled",
        "-Dvalgrind=disabled",
        "-Dtests=false",
        "-Dudev=false",
    )


def build_mesa_radv(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    """Build only the RADV Vulkan driver, without X11/Wayland or LLVM."""
    meson_install(
        source_dir,
        build_dir,
        install_dir,
        "-Dcpp_link_args=-Wl,-rpath,$ORIGIN",
        "-Dvulkan-drivers=amd",
        "-Dgallium-drivers=",
        "-Dplatforms=",
        "-Dglx=disabled",
        "-Degl=disabled",
        "-Dgbm=disabled",
        "-Dllvm=disabled",
        "-Dzstd=disabled",
        "-Dtools=",
        "-Dbuild-tests=false",
        "-Dvalgrind=disabled",
        "-Dlibunwind=disabled",
    )


def build_glslc(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    cmake_configure(
        source_dir,
        build_dir,
        install_dir,
        "-DSHADERC_SKIP_TESTS=ON",
        "-DSHADERC_SKIP_EXAMPLES=ON",
        "-DSHADERC_SKIP_COPYRIGHT_CHECK=ON",
    )
    run(["cmake", "--build", build_dir, "--target", "glslc_exe"])
    glslc = build_dir / "glslc" / "glslc"
    if not glslc.exists():
        raise SystemExit(f"Shaderc build did not produce glslc executable: {glslc}")
    bin_dir = install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(glslc, bin_dir / "glslc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_BUILD_ROOT / "vulkan-sdk-src",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_ROOT / "vulkan-sdk-build",
    )
    parser.add_argument(
        "--vulkan-sdk-dir",
        "--install-dir",
        dest="vulkan_sdk_dir",
        type=Path,
        default=DEFAULT_BUILD_ROOT / "vulkan-sdk",
        help="Vulkan SDK install root",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    install_dir = args.vulkan_sdk_dir.resolve()

    headers_src = source_dir / "Vulkan-Headers"
    loader_src = source_dir / "Vulkan-Loader"
    shaderc_src = source_dir / "shaderc"
    libdrm_src = source_dir / "drm"
    mesa_src = source_dir / "mesa"

    clone_or_update(VULKAN_HEADERS_REPO, VULKAN_HEADERS_TAG, headers_src)
    clone_or_update(VULKAN_LOADER_REPO, VULKAN_LOADER_TAG, loader_src)
    clone_or_update(SHADERC_REPO, SHADERC_TAG, shaderc_src)
    clone_or_update(LIBDRM_REPO, LIBDRM_TAG, libdrm_src)
    clone_or_update(MESA_REPO, MESA_TAG, mesa_src)
    sync_shaderc_deps(shaderc_src)

    build_vulkan_headers(headers_src, build_dir / "Vulkan-Headers", install_dir)
    build_vulkan_loader(loader_src, build_dir / "Vulkan-Loader", install_dir)
    build_spirv_headers(
        shaderc_src / "third_party" / "spirv-headers",
        build_dir / "SPIRV-Headers",
        install_dir,
    )
    build_glslc(shaderc_src, build_dir / "shaderc", install_dir)
    build_glslang_standalone(
        shaderc_src / "third_party" / "glslang",
        build_dir / "glslang",
        install_dir,
    )
    build_libdrm(libdrm_src, build_dir / "drm", install_dir)
    build_mesa_radv(mesa_src, build_dir / "mesa", install_dir)

    spirv_headers_dir = install_dir / "share" / "cmake" / "SPIRV-Headers"
    required = [
        install_dir / "include" / "vulkan" / "vulkan.h",
        install_dir / "include" / "spirv" / "unified1" / "spirv.hpp",
        install_dir / "bin" / "glslc",
        install_dir / "lib" / "libvulkan.so",
        install_dir / "lib" / "libvulkan.so.1",
        install_dir / "lib" / "libdrm.so.2",
        install_dir / "lib" / "libdrm_amdgpu.so.1",
        install_dir / "lib" / "libvulkan_radeon.so",
        install_dir / "share" / "vulkan" / "icd.d" / "radeon_icd.x86_64.json",
        spirv_headers_dir / "SPIRV-HeadersConfig.cmake",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Vulkan SDK build did not produce required files:\n  "
            + "\n  ".join(os.fspath(path) for path in missing)
        )
    run([install_dir / "bin" / "glslc", "--version"])
    print(f"Vulkan SDK installed to {install_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
