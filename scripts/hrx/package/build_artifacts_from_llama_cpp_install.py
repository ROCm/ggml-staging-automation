#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Create a release archive from a llama.cpp install tree."""

from __future__ import annotations

import argparse
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LLAMA_INSTALL_DIR = REPO_ROOT / "build" / "llama.cpp-install"
RELEASE_INSTALL_GLOBS = (
    "bin/llama-*",
    "include/*.h",
    "lib/**/*.cmake",
    "lib/**/*.pc",
)

RELEASE_REPO_FILES = (
    "LICENSE",
    "README.md",
)

RELEASE_LIBRARY_GLOBS = (
    "lib/**/*.so*",
)

@dataclass(frozen=True)
class PackageEntry:
    relative: PurePosixPath
    source: Path


def repo_entry(entry: str) -> PackageEntry:
    relative = PurePosixPath(entry)
    source = REPO_ROOT.joinpath(*relative.parts)
    return PackageEntry(relative=relative, source=source)


def glob_install_tree(install_dir: Path, pattern: str) -> list[PackageEntry]:
    matches = sorted(
        path
        for path in install_dir.glob(pattern)
        if path.is_file() or path.is_symlink()
    )
    if not matches:
        raise SystemExit(
            f"{install_dir} is missing package allow-list glob: {pattern}"
        )

    entries: list[PackageEntry] = []
    for source in matches:
        relative = PurePosixPath(source.relative_to(install_dir).as_posix())
        entries.append(PackageEntry(relative=relative, source=source))
    return entries


def release_entries(install_dir: Path) -> list[PackageEntry]:
    entries: list[PackageEntry] = []
    for pattern in RELEASE_INSTALL_GLOBS:
        entries.extend(glob_install_tree(install_dir, pattern))
    entries.extend(repo_entry(entry) for entry in RELEASE_REPO_FILES)
    for pattern in RELEASE_LIBRARY_GLOBS:
        entries.extend(glob_install_tree(install_dir, pattern))
    return entries


def normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip host-specific archive metadata for reproducible tar entries."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def archive_path(package_root: PurePosixPath, relative: PurePosixPath) -> str:
    return (package_root / relative).as_posix()


def create_package(
    *,
    package_file: Path,
    package_root: PurePosixPath,
    entries: Iterable[PackageEntry],
) -> list[str]:
    package_file = package_file.resolve()
    package_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = package_file.with_name(package_file.name + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    archived: list[str] = []
    try:
        with tarfile.open(tmp_file, mode="w:gz") as tar:
            for entry in entries:
                arcname = archive_path(package_root, entry.relative)
                tar.add(
                    entry.source,
                    arcname=arcname,
                    recursive=False,
                    filter=normalize_tarinfo,
                )
                archived.append(arcname)
        tmp_file.replace(package_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    return archived


def print_created(package_file: Path, entries: Iterable[str]) -> None:
    print(f"created {package_file.resolve()}", flush=True)
    for entry in entries:
        print(f"  {entry}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-install-dir",
        type=Path,
        default=DEFAULT_LLAMA_INSTALL_DIR,
    )
    parser.add_argument("--release-package-file", type=Path, required=True)
    parser.add_argument("--package-root-name", default="llama.cpp-install")
    args = parser.parse_args()

    install_dir = args.llama_install_dir.resolve()
    if not install_dir.exists():
        raise SystemExit(f"Missing llama.cpp install tree: {install_dir}")
    package_root = PurePosixPath(args.package_root_name)

    release = release_entries(install_dir)

    release_archived = create_package(
        package_file=args.release_package_file,
        package_root=package_root,
        entries=release,
    )
    print_created(args.release_package_file, release_archived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
