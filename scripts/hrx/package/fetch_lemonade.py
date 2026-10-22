#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Install the checksum-pinned embeddable Lemonade release without a source build."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def install(pin_path: Path, output: Path, archive: Path | None = None):
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    url = f"https://github.com/lemonade-sdk/lemonade/releases/download/{pin['tag']}/{pin['asset']}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary)
        if archive is None:
            archive = root / pin["asset"]
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pin["sha256"]:
            raise RuntimeError("Lemonade archive checksum mismatch")
        extracted = root / "unpacked"
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
        package = extracted / pin["asset"].removesuffix(".tar.gz")
        for file in ("lemonade", "lemond", "resources/defaults.json"):
            if not (package / file).is_file():
                raise RuntimeError(f"Release is missing {file}")
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(package), output)
    print(f"Installed Lemonade {pin['tag']} ({digest}) at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive")
    args = parser.parse_args()
    install(args.pin, args.output_dir, args.archive)
