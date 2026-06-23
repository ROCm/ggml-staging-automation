#!/usr/bin/env python3
"""Validate HRX/ROCm runtime library layout for llama.cpp build/install trees."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from hrx_build import add_common_path_args


REQUIRED_DEPS = (
    "libhrx.so",
)

BUNDLED_FILES = (
    "libhrx.so",
    "libloomc.so",
    "libhsa-runtime64.so",
    "libhsa-amd-aqlprofile64.so",
    "librocprofiler-register.so",
    "libomp.so",
)


def command_output(args: list[str | os.PathLike[str]]) -> str:
    result = subprocess.run(
        [os.fspath(arg) for arg in args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def dynamic_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for subdir in ("bin", "lib"):
        directory = root / subdir
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and is_elf(path):
                candidates.append(path)
    return sorted(candidates)


def ldd_paths(binary: Path) -> dict[str, Path]:
    output = command_output(["ldd", binary])
    deps: dict[str, Path] = {}
    for line in output.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)\s+\(", line)
        if match:
            deps[match.group(1)] = Path(match.group(2)).resolve()
    return deps


def rpath_text(binary: Path) -> str:
    output = command_output(["readelf", "-d", binary])
    return "\n".join(
        line for line in output.splitlines() if "RPATH" in line or "RUNPATH" in line
    )


def validate_tree(root: Path, rocm_root: Path) -> None:
    if not root.exists():
        raise SystemExit(f"Missing tree: {root}")
    files = dynamic_files(root)
    if not files:
        raise SystemExit(f"No ELF files found under {root}/bin or {root}/lib")

    all_deps: dict[str, Path] = {}
    rpath_checked = False
    for binary in files:
        try:
            text = rpath_text(binary)
        except subprocess.CalledProcessError:
            text = ""
        if "$ORIGIN" in text:
            rpath_checked = True
        try:
            all_deps.update(ldd_paths(binary))
        except subprocess.CalledProcessError:
            continue

    missing = [dep for dep in REQUIRED_DEPS if not any(name.startswith(dep) for name in all_deps)]
    if missing:
        raise SystemExit(
            f"{root} did not expose expected direct HRX dependencies via ldd:\n  "
            + "\n  ".join(missing)
        )
    bundle_roots = [path for path in (root / "bin", root / "lib", root) if path.exists()]
    missing_bundled = [
        filename
        for filename in BUNDLED_FILES
        if not any((base / filename).exists() for base in bundle_roots)
    ]
    if missing_bundled:
        raise SystemExit(
            f"{root} did not contain expected bundled ROCm runtime files:\n  "
            + "\n  ".join(missing_bundled)
        )

    escaped = rocm_root.resolve()
    bad = [
        f"{name} => {path}"
        for name, path in sorted(all_deps.items())
        if escaped == path or escaped in path.parents
    ]
    if bad:
        raise SystemExit(
            f"{root} resolves bundled dependencies from the ROCm build root:\n  "
            + "\n  ".join(bad)
        )
    if not rpath_checked:
        raise SystemExit(f"No $ORIGIN RPATH/RUNPATH found in ELF files under {root}")
    print(f"validated {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    args = parser.parse_args()
    validate_tree(args.llama_build_dir.resolve(), args.rocm_root.resolve())
    validate_tree(args.llama_install_dir.resolve(), args.rocm_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
