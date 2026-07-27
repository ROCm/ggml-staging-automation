#!/usr/bin/env python3
"""Fetch the pinned minimal ROCm build root from TheRock artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys

from hrx_build import (
    REPO_ROOT,
    ROCM_ARTIFACT_SET,
    ROCM_EXTRA_BUILD_ARTIFACTS,
    add_common_path_args,
    python_executable,
    read_rocm_pin,
    require_submodule,
    run,
)


def fetch_extra_build_artifacts(
    *,
    hrx_system,
    release_type: str,
    run_id: str,
    rocm_root,
    download_cache_dir,
) -> None:
    if not ROCM_EXTRA_BUILD_ARTIFACTS:
        return

    sys.path.insert(0, os.fspath(hrx_system))
    from build_tools import ci_core_common as ci

    s3 = ci.create_s3_client()
    bucket = ci.release_bucket(release_type, "artifacts")
    prefix = f"{run_id}-linux/"
    available = ci.list_prefix(s3, bucket, prefix)
    selected, missing = ci.select_available(
        available, prefix, list(ROCM_EXTRA_BUILD_ARTIFACTS)
    )
    if missing:
        raise SystemExit(
            "Missing required build-only ROCm artifacts:\n  " + "\n  ".join(missing)
        )

    print("Build-only artifacts selected:")
    for obj in selected:
        print(f"  {obj.key} ({obj.size / 1024 / 1024:.1f} MiB)", flush=True)

    rocm_root = rocm_root.resolve()
    download_cache_dir = download_cache_dir.resolve()
    download_cache_dir.mkdir(parents=True, exist_ok=True)
    for obj in selected:
        archive_path = ci.download_one(s3, bucket, obj, download_cache_dir)
        checksum = ci.download_checksum(s3, bucket, obj.key, archive_path)
        ci.verify_checksum(archive_path, checksum)
        print(f"  ++ Flattening {archive_path.name}", flush=True)
        ci.flatten_therock_artifact(archive_path, rocm_root)

    manifest_path = rocm_root / ".hrx-rocm-artifacts.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest["extra_build_artifacts"] = [obj.__dict__ for obj in selected]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--download-concurrency", type=int, default=8)
    args = parser.parse_args()

    hrx_system = REPO_ROOT / "hrx-system"
    require_submodule(hrx_system)
    pin = read_rocm_pin()

    run(
        [
            python_executable(),
            "build_tools/ci_core_linux.py",
            "fetch-rocm",
            "--release-type",
            pin["release_type"],
            "--run-id",
            pin["run_id"],
            "--artifact-set",
            ROCM_ARTIFACT_SET,
            "--rocm-root",
            args.rocm_root.resolve(),
            "--download-cache-dir",
            args.download_cache_dir.resolve(),
            "--download-concurrency",
            str(args.download_concurrency),
        ],
        cwd=hrx_system,
    )
    fetch_extra_build_artifacts(
        hrx_system=hrx_system,
        release_type=pin["release_type"],
        run_id=pin["run_id"],
        rocm_root=args.rocm_root,
        download_cache_dir=args.download_cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
