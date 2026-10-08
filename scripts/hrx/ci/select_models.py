#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Select CI coverage from GitHub's event payload.

PRs opt into the cumulative full model tier with ``ci:full-models``. Adding
that label starts CI; unrelated label events do not start builds or GPU jobs.
Removing it takes effect on the next normal PR update. Pushes use full, and
manual runs use their model_tier input (default smoke). The CI workflow
consumes should_run and model_tier through GITHUB_OUTPUT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FULL_MODELS_LABEL = "ci:full-models"


def select_models(event_name: str, event: dict) -> tuple[bool, str]:
    if event_name == "pull_request":
        is_label_event = event["action"] == "labeled"
        is_full_models_label = event.get("label", {}).get("name") == FULL_MODELS_LABEL
        is_unrelated_label = is_label_event and not is_full_models_label
        labels = {label["name"] for label in event["pull_request"]["labels"]}
        model_tier = "full" if FULL_MODELS_LABEL in labels else "smoke"
        return not is_unrelated_label, model_tier

    if event_name == "workflow_dispatch":
        return True, event["inputs"]["model_tier"]

    return True, "full"


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    should_run, model_tier = select_models(os.environ["GITHUB_EVENT_NAME"], event)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"should_run={str(should_run).lower()}\n")
        output.write(f"model_tier={model_tier}\n")
    print(f"CI coverage: should_run={should_run}, model_tier={model_tier}")


if __name__ == "__main__":
    main()
