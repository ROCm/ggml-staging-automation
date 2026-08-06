#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check whether the current source commit needs a release."""

from __future__ import annotations

import json
import os
import subprocess


def gh(*args: str):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def report(should_publish: bool, message: str, *details: tuple[str, str]) -> int:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"should_publish={str(should_publish).lower()}\n")

    print("### Release source check\n")
    for label, value in details:
        print(f"- {label}: `{value}`")
    if details:
        print()
    print(message)
    return 0


def main() -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    current_sha = os.environ["GITHUB_SHA"]
    releases = gh(
        "release",
        "list",
        "--repo",
        repository,
        "--exclude-drafts",
        "--limit",
        "1",
        "--json",
        "tagName",
    )

    if not releases:
        return report(
            True, "No existing release was found; publishing the initial release."
        )

    latest_tag = releases[0]["tagName"]
    release = gh(
        "release",
        "view",
        latest_tag,
        "--repo",
        repository,
        "--json",
        "targetCommitish",
    )
    released_sha = release["targetCommitish"]
    changed = current_sha.lower() != released_sha.lower()
    decision = (
        "Source changed; continuing with the release."
        if changed
        else "No source change; skipping build, tests, and publishing."
    )
    return report(
        changed,
        decision,
        ("Latest release", latest_tag),
        ("Released source commit", released_sha),
        ("Current source commit", current_sha),
    )


if __name__ == "__main__":
    raise SystemExit(main())
