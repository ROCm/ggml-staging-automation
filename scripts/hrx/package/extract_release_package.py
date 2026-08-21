#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Extract a release package to a stable directory."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--package-root-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists():
        parser.error(f"--output-dir already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
            with tarfile.open(args.archive, mode="r:gz") as archive:
                archive.extractall(temporary, filter="data")

            extracted_root = Path(temporary) / args.package_root_name
            if not extracted_root.is_dir():
                raise RuntimeError(
                    f"Archive is missing directory {args.package_root_name}"
                )
            extracted_root.rename(output_dir)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(1, f"Release package extraction failed: {exc}\n")

    print(f"Extracted {args.archive} to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
