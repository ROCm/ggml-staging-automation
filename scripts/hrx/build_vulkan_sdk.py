#!/usr/bin/env python3
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


def build_glslc(source_dir: Path, build_dir: Path, install_dir: Path) -> None:
    run(["python3", "utils/git-sync-deps"], cwd=source_dir)
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

    clone_or_update(VULKAN_HEADERS_REPO, VULKAN_HEADERS_TAG, headers_src)
    clone_or_update(VULKAN_LOADER_REPO, VULKAN_LOADER_TAG, loader_src)
    clone_or_update(SHADERC_REPO, SHADERC_TAG, shaderc_src)

    build_vulkan_headers(headers_src, build_dir / "Vulkan-Headers", install_dir)
    build_vulkan_loader(loader_src, build_dir / "Vulkan-Loader", install_dir)
    build_glslc(shaderc_src, build_dir / "shaderc", install_dir)

    vulkan_loader = next(
        (
            path
            for path in (
                install_dir / "lib" / "libvulkan.so",
                install_dir / "lib64" / "libvulkan.so",
            )
            if path.exists()
        ),
        None,
    )
    required = [
        install_dir / "include" / "vulkan" / "vulkan.h",
        install_dir / "bin" / "glslc",
    ]
    missing = [path for path in required if not path.exists()]
    if vulkan_loader is None:
        missing.append(install_dir / "lib" / "libvulkan.so")
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
