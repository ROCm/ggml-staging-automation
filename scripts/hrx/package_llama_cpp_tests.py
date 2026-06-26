#!/usr/bin/env python3
"""Package selected llama.cpp tests as an overlay tar.gz archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from hrx_build import add_common_path_args


DEFAULT_PACKAGE_NAME = "llama-cpp-linux-x86_64-tests.tar.gz"
TEST_PACKAGE_ALLOWLIST = ("bin/test-backend-ops",)


def validate_allowlist_entry(entry: str) -> PurePosixPath:
    path = PurePosixPath(entry)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"Invalid package allow-list entry: {entry}")
    return path


def validate_package_root_name(package_root_name: str) -> PurePosixPath:
    return validate_allowlist_entry(package_root_name)


def package_entries(
    install_dir: Path,
    allowlist: Iterable[str],
) -> list[tuple[PurePosixPath, Path]]:
    entries: list[tuple[PurePosixPath, Path]] = []
    missing: list[str] = []
    for entry in allowlist:
        relative = validate_allowlist_entry(entry)
        source = install_dir.joinpath(*relative.parts)
        if not source.is_file():
            missing.append(entry)
            continue
        entries.append((relative, source))
    if missing:
        raise SystemExit(
            f"{install_dir} is missing package allow-list entries:\n  "
            + "\n  ".join(missing)
        )
    return entries


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def stage_package_entries(
    *,
    staging_root: Path,
    entries: Iterable[tuple[PurePosixPath, Path]],
) -> None:
    remove_path(staging_root)
    staging_root.mkdir(parents=True)
    for relative, source in entries:
        destination = staging_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def create_test_package(
    *,
    install_dir: Path,
    package_file: Path,
    package_root_name: str,
    allowlist: Iterable[str] = TEST_PACKAGE_ALLOWLIST,
) -> list[str]:
    install_dir = install_dir.resolve()
    package_file = package_file.resolve()
    if not install_dir.exists():
        raise SystemExit(f"Missing llama.cpp test install tree: {install_dir}")

    package_root = validate_package_root_name(package_root_name)
    entries = package_entries(install_dir, allowlist)
    package_file.parent.mkdir(parents=True, exist_ok=True)
    staging_root = package_file.parent.joinpath(*package_root.parts)
    if staging_root.resolve() == install_dir:
        raise SystemExit(
            "Package staging root must not be the same as the llama.cpp test install tree: "
            f"{staging_root}"
        )
    stage_package_entries(staging_root=staging_root, entries=entries)

    tmp_file = package_file.with_name(package_file.name + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    try:
        with tarfile.open(tmp_file, mode="w:gz") as tar:
            tar.add(
                staging_root,
                arcname=package_root.as_posix(),
                recursive=True,
                filter=normalize_tarinfo,
            )
        tmp_file.replace(package_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    return [(package_root / relative).as_posix() for relative, _ in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument(
        "--package-file",
        type=Path,
        default=None,
        help=f"default: <package-dir>/{DEFAULT_PACKAGE_NAME}",
    )
    parser.add_argument("--package-root-name", default="llama.cpp-install")
    args = parser.parse_args()

    package_file = args.package_file or args.package_dir / DEFAULT_PACKAGE_NAME
    entries = create_test_package(
        install_dir=args.llama_test_install_dir,
        package_file=package_file,
        package_root_name=args.package_root_name,
    )
    print(f"created {package_file.resolve()}", flush=True)
    for entry in entries:
        print(f"  {entry}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
