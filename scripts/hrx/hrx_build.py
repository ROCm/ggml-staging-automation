#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared helpers for local HRX/llama.cpp build scripts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
ROCM_PIN_FILE = REPO_ROOT / "rocm-version.json"
ROCM_ARTIFACT_SET = "core-with-upstream-hip"
ROCM_EXTRA_BUILD_ARTIFACTS = ("rocwmma_dev_generic",)
AMDGPU_TARGETS = ("gfx1100", "gfx1151", "gfx1201")


def default_paths() -> dict[str, Path]:
    build_root = REPO_ROOT / "build"
    return {
        "build_root": build_root,
        "downloads": build_root / "downloads",
        "rocm_root": build_root / "rocm-root",
        "hrx_build": build_root / "hrx-system-build",
        "hrx_install": build_root / "hrx-system-install",
        "llama_build": build_root / "llama.cpp-build",
        "llama_install": build_root / "llama.cpp-install",
        "packages": build_root / "packages",
    }


def add_common_path_args(parser: argparse.ArgumentParser) -> None:
    paths = default_paths()
    parser.add_argument("--rocm-root", type=Path, default=paths["rocm_root"])
    parser.add_argument("--download-cache-dir", type=Path, default=paths["downloads"])
    parser.add_argument("--hrx-build-dir", type=Path, default=paths["hrx_build"])
    parser.add_argument("--hrx-install-dir", type=Path, default=paths["hrx_install"])
    parser.add_argument("--llama-build-dir", type=Path, default=paths["llama_build"])
    parser.add_argument("--llama-install-dir", type=Path, default=paths["llama_install"])
    parser.add_argument("--package-dir", type=Path, default=paths["packages"])


def targets_cmake_list() -> str:
    return ";".join(AMDGPU_TARGETS)


def run(
    args: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [os.fspath(arg) for arg in args]
    print("++", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def require_submodule(path: Path) -> None:
    if not (path / "CMakeLists.txt").exists():
        raise SystemExit(
            f"Missing initialized submodule at {path}. "
            "Run `git submodule update --init hrx-system llama.cpp`."
        )


def read_rocm_pin(pin_file: Path = ROCM_PIN_FILE) -> dict[str, str]:
    if not pin_file.exists():
        raise SystemExit(f"Missing ROCm pin file: {pin_file}")
    with pin_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    release_type = str(data.get("release_type") or "nightly")
    run_id = str(data.get("run_id") or "").strip()
    if release_type not in {"dev", "nightly", "prerelease"}:
        raise SystemExit(
            f"{pin_file} release_type must be dev, nightly, or prerelease"
        )
    if not run_id:
        raise SystemExit(
            f"{pin_file} must contain a non-empty TheRock run_id. "
            "The default release_type is nightly."
        )
    return {"release_type": release_type, "run_id": run_id}


def rocm_tool(rocm_root: Path, name: str) -> Path:
    tool = rocm_root / "lib" / "llvm" / "bin" / name
    if not tool.exists():
        raise SystemExit(f"Missing ROCm LLVM tool {name}: {tool}")
    return tool


def require_rocm_root(rocm_root: Path) -> None:
    required = [
        rocm_root / "lib" / "llvm" / "bin" / "clang",
        rocm_root / "lib" / "llvm" / "bin" / "clang++",
        rocm_root / "lib" / "libhsa-runtime64.so",
        rocm_root / "lib" / "libhsa-amd-aqlprofile64.so",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "ROCm root is incomplete. Missing:\n  "
            + "\n  ".join(os.fspath(path) for path in missing)
        )


def remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def rocm_env(rocm_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    llvm_bin = rocm_root / "lib" / "llvm" / "bin"
    prepend_path(env, "PATH", [llvm_bin, rocm_root / "bin"])
    prepend_path(env, "LD_LIBRARY_PATH", [rocm_root / "lib", rocm_root / "lib" / "rocm_sysdeps" / "lib"])
    prepend_path(env, "CMAKE_PREFIX_PATH", [rocm_root], separator=os.pathsep)
    env["CC"] = os.fspath(rocm_tool(rocm_root, "clang"))
    env["CXX"] = os.fspath(rocm_tool(rocm_root, "clang++"))
    return env


def prepend_path(
    env: dict[str, str],
    name: str,
    paths: Iterable[Path],
    *,
    separator: str = os.pathsep,
) -> None:
    existing = env.get(name, "")
    parts = [os.fspath(path) for path in paths if path.exists()]
    if existing:
        parts.append(existing)
    env[name] = separator.join(parts)


def maybe_cmake_launcher_args() -> list[str]:
    if shutil.which("ccache"):
        return [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    return []


def cmake_toolchain_args(rocm_root: Path) -> list[str]:
    return [
        f"-DCMAKE_C_COMPILER={rocm_tool(rocm_root, 'clang')}",
        f"-DCMAKE_CXX_COMPILER={rocm_tool(rocm_root, 'clang++')}",
        f"-DCMAKE_ASM_COMPILER={rocm_tool(rocm_root, 'clang')}",
        f"-DCMAKE_AR={rocm_tool(rocm_root, 'llvm-ar')}",
        f"-DCMAKE_RANLIB={rocm_tool(rocm_root, 'llvm-ranlib')}",
        *maybe_cmake_launcher_args(),
    ]


def cmake_generator_args() -> list[str]:
    return ["-GNinja"] if shutil.which("ninja") else []


def cmake_build_type_arg(build_type: str) -> str:
    return f"-DCMAKE_BUILD_TYPE={build_type}"


def python_executable() -> str:
    return sys.executable or "python3"


def main() -> int:
    print("This module is shared by the scripts in scripts/hrx.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
