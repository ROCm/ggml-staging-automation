#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check coverage policy and whole-PR diffs without running GPU workloads."""

import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import select_models as coverage


def event(labels=(), action="synchronize", added=""):
    return {"action": action, "label": {"name": added},
            "pull_request": {"labels": [{"name": name} for name in labels]}}


class CoverageTests(unittest.TestCase):
    def test_path_policy(self):
        relevant = [
            "llama.cpp", "hrx-system", ".gitmodules", "requirements.txt", "rocm-version.json",
            "scripts/hrx/build/build_all.py", "scripts/hrx/benchmark/run_perplexity_benchmark.py",
            "scripts/hrx/package/extract_release_package.py", "scripts/hrx/ci/select_models.py",
            "benchmarks/hrx/model_manifest.json", ".github/workflows/ci.yml",
            ".github/workflows/build_llama_cpp_linux.yml", ".github/workflows/release_test_bench.yml",
        ]
        excluded = [
            "README.md", "SECURITY.md", "LICENSE", ".github/CODEOWNERS", ".github/dependabot.yml",
            "zizmor.yml", "bandit.yaml", "gitleaks.toml", "scan_tools/github_actions/zizmor.py",
            ".github/workflows/zizmor.yml", ".github/workflows/zizmor_main.yml",
            ".github/workflows/security.yml", ".github/workflows/codeql.yml",
            ".github/workflows/bandit.yml", ".github/workflows/release.yml",
            ".github/workflows/bump_submodules.yml", "scripts/hrx/bump/update_submodules.py",
            "scripts/hrx/release/check_release_source.py",
        ]
        cases = [([path], path in relevant) for path in relevant + excluded]
        cases.extend([([], False), (["README.md", "llama.cpp"], True)])
        for paths, expected in cases:
            with self.subTest(paths=paths), patch.object(coverage, "changed_paths", return_value=paths):
                self.assertEqual(coverage.select_models("pull_request", event())[:2], (expected, "single"))

    def test_labels_and_manual_dispatch_bypass_diff(self):
        cases = [(("ci:single-model",), "single"), (("ci:small-models",), "small"),
                 (("ci:all-models",), "all"), (("ci:small-models", "ci:single-model"), "small"),
                 (("ci:all-models", "ci:single-model", "ci:small-models"), "all")]
        with patch.object(coverage, "changed_paths", side_effect=AssertionError("unexpected diff")):
            for labels, tier in cases:
                for action in ("opened", "synchronize", "reopened", "labeled"):
                    with self.subTest(labels=labels, action=action):
                        result = coverage.select_models("pull_request", event(labels, action, labels[-1]))
                        self.assertEqual(result[:2], (True, tier))
            result = coverage.select_models("pull_request", event(("ci:all-models",), "labeled", "bug"))
            self.assertEqual(result[:2], (False, "all"))
            for tier in ("single", "small", "all"):
                result = coverage.select_models("workflow_dispatch", {"inputs": {"model_tier": tier}})
                self.assertEqual(result[:2], (True, tier))

    def test_diff_failure_surfaces(self):
        with patch.object(coverage, "changed_paths", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(subprocess.CalledProcessError):
                coverage.select_models("pull_request", event())

    def test_whole_pr_diff_and_move_out_of_relevant_paths(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            def git(*args):
                return subprocess.check_output(["git", *args], text=True).strip()

            def commit(message):
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", message)
                return git("rev-parse", "HEAD")

            git("init", "-q")
            Path("requirements.txt").write_text("initial\n")
            base = commit("base")
            Path("requirements.txt").write_text("changed\n")
            commit("relevant change")
            Path("README.md").write_text("docs\n")
            head = commit("docs only")
            payload = event()
            payload["pull_request"].update(base={"sha": base}, head={"sha": head})
            self.assertEqual(set(coverage.changed_paths(payload)), {"requirements.txt", "README.md"})
            self.assertTrue(coverage.select_models("pull_request", payload)[0])
            Path("requirements.txt").rename("notes.txt")
            payload["pull_request"]["head"]["sha"] = commit("move out of relevant paths")
            self.assertIn("requirements.txt", coverage.changed_paths(payload))
            self.assertTrue(coverage.select_models("pull_request", payload)[0])

    def test_output_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "event.json"
            output = Path(directory) / "output"
            payload.write_text(json.dumps({"inputs": {"model_tier": "all"}}))
            env = {"GITHUB_EVENT_PATH": str(payload), "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_OUTPUT": str(output)}
            with patch.dict(os.environ, env):
                coverage.main()
            self.assertEqual(output.read_text(), "should_run=true\nmodel_tier=all\n")


if __name__ == "__main__":
    unittest.main()
