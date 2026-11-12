# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Record the checked-out automation, llama.cpp, and HRX source revisions."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def source_revisions() -> dict[str, str]:
    """Return exact revisions used to build and run the benchmark tools."""
    revisions = {}
    for name, path in (
        ("automation", REPOSITORY_ROOT),
        ("llama_cpp", REPOSITORY_ROOT / "llama.cpp"),
        ("hrx_system", REPOSITORY_ROOT / "hrx-system"),
    ):
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    return revisions
