#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Record source revisions so a persisted benchmark remains attributable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    parser.add_argument("--hrx-system-commit", required=True)
    parser.add_argument("--lemonade-release", type=Path, required=True)
    args = parser.parse_args()

    metadata = {
        "schema_version": 2,
        "llama_cpp": args.llama_cpp_commit,
        "hrx_system": args.hrx_system_commit,
        "lemonade": json.loads(args.lemonade_release.read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote benchmark metadata to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
