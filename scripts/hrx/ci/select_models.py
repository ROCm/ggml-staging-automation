#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Select CI coverage from GitHub's event payload.

Coverage labels and manual dispatch override path filtering. Unlabeled PRs
use the single tier when their whole diff affects build or benchmark inputs.
The CI workflow consumes should_run and model_tier through GITHUB_OUTPUT;
README.md describes the coverage policy and label behavior.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

MODEL_LABELS = {
    "ci:single-model": "single",
    "ci:small-models": "small",
    "ci:all-models": "all",
}
RELEVANT_FILES = {
    "llama.cpp",
    "hrx-system",
    ".gitmodules",
    "requirements.txt",
    "rocm-version.json",
    ".github/workflows/ci.yml",
    ".github/workflows/build_llama_cpp_linux.yml",
    ".github/workflows/release_test_bench.yml",
}
RELEVANT_DIRECTORIES = (
    "scripts/hrx/build/",
    "scripts/hrx/benchmark/",
    "scripts/hrx/package/",
    "scripts/hrx/ci/",
    "benchmarks/hrx/",
)


def changed_paths(event: dict) -> list[str]:
    """Compare the complete PR, including deleted paths and both sides of moves."""
    pr = event["pull_request"]
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z",
         f"{pr['base']['sha']}...{pr['head']['sha']}", "--"],
    )
    return [os.fsdecode(path) for path in diff.split(b"\0") if path]


def is_relevant(path: str) -> bool:
    is_input_file = path in RELEVANT_FILES
    is_input_directory = path.startswith(RELEVANT_DIRECTORIES)
    return is_input_file or is_input_directory


def select_models(event_name: str, event: dict) -> tuple[bool, str, str]:
    if event_name == "workflow_dispatch":
        return True, event["inputs"]["model_tier"], "manual dispatch"

    if event_name == "pull_request":
        is_label_event = event["action"] == "labeled"
        is_coverage_label = event.get("label", {}).get("name") in MODEL_LABELS
        is_unrelated_label = is_label_event and not is_coverage_label
        labels = {label["name"] for label in event["pull_request"]["labels"]}
        requested_tiers = [tier for label, tier in MODEL_LABELS.items() if label in labels]
        model_tier = requested_tiers[-1] if requested_tiers else "single"
        if is_unrelated_label:
            return False, model_tier, "unrelated label event"
        if requested_tiers:
            return True, model_tier, "explicit coverage label"
        if any(is_relevant(path) for path in changed_paths(event)):
            return True, model_tier, "build or benchmark inputs changed"
        return False, model_tier, "no build or benchmark inputs changed"

    raise ValueError(f"Unsupported CI event: {event_name}")


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    should_run, model_tier, reason = select_models(os.environ["GITHUB_EVENT_NAME"], event)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"should_run={str(should_run).lower()}\n")
        output.write(f"model_tier={model_tier}\n")
    print(f"CI coverage: should_run={should_run}, model_tier={model_tier}: {reason}")


if __name__ == "__main__":
    main()
