#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Contract checks for long-context statuses and CLI validation."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_outcomes import aggregate_status, classify_outcome
import run_context_sweep
import run_perplexity_benchmark
from run_batched_benchmark import load_manifest


class LongContextOutcomeTests(unittest.TestCase):
    def test_unsupported_coverage_is_a_failure_not_a_pass_or_xpass(self) -> None:
        self.assertEqual(aggregate_status(["unsupported"]), "fail")
        self.assertEqual(aggregate_status(["ok", "unsupported"]), "fail")
        self.assertEqual(classify_outcome("fail", "pass"), "FAIL")
        self.assertEqual(classify_outcome("fail", "fail"), "XFAIL")
        self.assertEqual(
            classify_outcome("fail", "fail", expected_failure_allowed=False), "FAIL"
        )

    def test_failures_take_precedence_over_unsupported_coverage(self) -> None:
        self.assertEqual(aggregate_status(["ok", "unsupported", "failed"]), "fail")
        self.assertEqual(classify_outcome("fail", "fail"), "XFAIL")
        self.assertEqual(classify_outcome("fail", "pass"), "FAIL")

    def test_manifest_accepts_explicit_context_modes_with_default_pass(self) -> None:
        manifest_path = Path(__file__).resolve().parents[4] / "benchmarks/hrx/model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest["models"][0]
        model["test-modes"] = {"context-sweep": ["65536"]}
        model["expected_results"] = {
            "context-sweep": {"hrx": {"65536": "fail"}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "manifest.json"
            fixture.write_text(json.dumps(manifest), encoding="utf-8")
            resolved = load_manifest(fixture).models[0]
        self.assertEqual(resolved.expectation("context-sweep", "hrx", "65536"), "fail")
        self.assertEqual(resolved.expectation("context-sweep", "hrx", "32768"), "pass")

    def test_context_sweep_rejects_empty_contexts_and_invalid_workload_sizes(self) -> None:
        for invalid in (
            ["--contexts", ""],
            ["--contexts", "32768", "--decode-tokens", "32768"],
            ["--contexts", "32768", "--batch-size", "0"],
            ["--contexts", "32768", "--run-timeout-seconds", "nan"],
        ):
            with self.subTest(invalid=invalid), patch.object(
                sys, "argv", ["run_context_sweep.py", *self._required_args(), *invalid]
            ), self.assertRaises(SystemExit) as raised:
                run_context_sweep.parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_long_ppl_rejects_empty_contexts_and_invalid_score_count(self) -> None:
        for invalid in (
            ["--context-lengths", ""],
            ["--context-lengths", "32768", "--scored-tokens", "32768"],
            ["--context-lengths", "32768", "--run-timeout-seconds", "0"],
        ):
            argv = [
                "run_perplexity_benchmark.py", "--stream-perplexity", "evaluator",
                "--model-manifest", "manifest", "--corpus-file", "corpus",
                "--models-dir", "models", "--output", "out",
                "--hrx-log", "hrx.log", "--vulkan-log", "vulkan.log",
                "--models", "model", *invalid,
            ]
            with self.subTest(invalid=invalid), patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
                run_perplexity_benchmark.parse_args()
            self.assertEqual(raised.exception.code, 2)

    @staticmethod
    def _required_args() -> list[str]:
        return [
            "--llama-bench", "llama-bench", "--context-inspector", "inspector",
            "--model-manifest", "manifest", "--models-dir", "models",
            "--output", "out", "--hrx-log", "hrx.log", "--vulkan-log", "vulkan.log",
            "--models", "model",
        ]


if __name__ == "__main__":
    unittest.main()
