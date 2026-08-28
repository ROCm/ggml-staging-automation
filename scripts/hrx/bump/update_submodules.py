#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Update the integration submodule pins and report the change.

Each submodule moves to the head of the branch named in ``.gitmodules``
unless an explicit 40-character commit SHA arrives through the environment
(``HRX_COMMIT`` / ``LLAMA_COMMIT``, the workflow's dispatch inputs).  The
old and new pins are appended to ``GITHUB_OUTPUT`` as ``<name>_from`` /
``<name>_to`` (plus 12-character ``_short`` forms) for the PR body.
"""

from __future__ import annotations

import os
import re
import subprocess

SUBMODULES = (
    ("hrx", "hrx-system", "HRX_COMMIT"),
    ("llama", "llama.cpp", "LLAMA_COMMIT"),
)


def git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def update_submodule(path: str, commit: str) -> None:
    if not commit:
        git("submodule", "update", "--remote", "--checkout", "--", path)
        return

    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise SystemExit(f"{path} commit must be a full 40-character SHA.")

    git("-C", path, "fetch", "origin", commit)
    git("-C", path, "checkout", "--detach", commit)


def main() -> int:
    paths = [path for _, path, _ in SUBMODULES]
    pins_from = {path: git_output("rev-parse", f"HEAD:{path}") for path in paths}

    git("submodule", "sync", "--", *paths)
    git("submodule", "update", "--init", "--checkout", "--", *paths)

    for _, path, commit_var in SUBMODULES:
        update_submodule(path, os.environ.get(commit_var, ""))

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for name, path, _ in SUBMODULES:
            pin_from = pins_from[path]
            pin_to = git_output("-C", path, "rev-parse", "HEAD")
            output.write(f"{name}_from={pin_from}\n")
            output.write(f"{name}_from_short={pin_from[:12]}\n")
            output.write(f"{name}_to={pin_to}\n")
            output.write(f"{name}_to_short={pin_to[:12]}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
