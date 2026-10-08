# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise CI event routing and the GitHub Actions output boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from select_models import select_models


class SelectModelsTests(unittest.TestCase):
    def test_pr_coverage_follows_labels_on_normal_updates(self) -> None:
        for action in ("opened", "synchronize", "reopened"):
            for labels, expected in (([], "smoke"), (["ci:full-models"], "full")):
                with self.subTest(action=action, labels=labels):
                    event = {
                        "action": action,
                        "pull_request": {"labels": [{"name": name} for name in labels]},
                    }
                    self.assertEqual(select_models("pull_request", event), (True, expected))

    def test_only_opt_in_label_starts_ci(self) -> None:
        for label, should_run in (("ci:full-models", True), ("documentation", False)):
            for existing_labels in ([], [{"name": "ci:full-models"}]):
                with self.subTest(label=label, existing_labels=existing_labels):
                    event = {
                        "action": "labeled",
                        "label": {"name": label},
                        "pull_request": {"labels": [*existing_labels, {"name": label}]},
                    }
                    self.assertEqual(select_models("pull_request", event)[0], should_run)

    def test_push_and_manual_tiers(self) -> None:
        self.assertEqual(select_models("push", {}), (True, "full"))
        for tier in ("smoke", "full"):
            with self.subTest(tier=tier):
                event = {"inputs": {"model_tier": tier, "skip_benchmarks": '["perplexity"]'}}
                self.assertEqual(select_models("workflow_dispatch", event), (True, tier))

    def test_entrypoint_appends_actions_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            output_path = Path(directory) / "output"
            event_path.write_text(json.dumps({
                "action": "labeled",
                "label": {"name": "ci:full-models"},
                "pull_request": {"labels": [{"name": "ci:full-models"}]},
            }), encoding="utf-8")
            output_path.write_text("existing=value\n", encoding="utf-8")
            environment = dict(os.environ)
            environment.update({
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_EVENT_PATH": str(event_path),
                "GITHUB_OUTPUT": str(output_path),
            })
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("select_models.py"))],
                env=environment, check=True, capture_output=True, text=True,
            )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "existing=value\nshould_run=true\nmodel_tier=full\n",
            )


if __name__ == "__main__":
    unittest.main()
