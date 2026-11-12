#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Contract checks for long-context statuses and CLI validation."""

from __future__ import annotations

import sys
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_outcomes import aggregate_status, classify_outcome
import run_context_sweep
import run_perplexity_benchmark
from run_batched_benchmark import load_manifest
from run_batched_benchmark import ModelSpec


class LongContextOutcomeTests(unittest.TestCase):
    def test_unsupported_coverage_is_a_failure_not_a_pass_or_xpass(self) -> None:
        self.assertEqual(aggregate_status(["unsupported"]), "fail")
        self.assertEqual(aggregate_status(["ok", "unsupported"]), "fail")
        self.assertEqual(classify_outcome("fail", "pass"), "FAIL")
        self.assertEqual(classify_outcome("fail", "fail"), "XFAIL")
        self.assertEqual(
            classify_outcome("fail", "fail", expected_failure_allowed=False), "FAIL"
        )
        self.assertEqual(
            classify_outcome("fail", "report-only", infrastructure_failure=True), "FAIL"
        )

    def test_failures_take_precedence_over_unsupported_coverage(self) -> None:
        self.assertEqual(aggregate_status(["ok", "unsupported", "failed"]), "fail")
        self.assertEqual(aggregate_status(["pass", "pass"]), "pass")
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

    def test_context_sweep_records_per_context_xfail_and_failure_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "models"
            model_path = models_dir / "Model" / "model.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.touch()
            llama_bench = root / "llama-bench"
            inspector = root / "inspector"
            llama_bench.touch()
            inspector.touch()
            spec = ModelSpec(
                id="model", tier="all",
                expected_results={"context-sweep": {
                    "hrx": {"32": "fail", "64": "fail"},
                    "vulkan": {"64": "report-only"},
                }},
                test_modes={"context-sweep": ("32", "64")},
                name="Model", repository="owner/model", revision="rev",
                filename="model.gguf", size_bytes=1, sha256="0" * 64,
            )
            args = argparse.Namespace(
                llama_bench=llama_bench, context_inspector=inspector,
                model_manifest=root / "manifest.json", models_dir=models_dir,
                models=["Model"], output=root / "out.json",
                hrx_log=root / "hrx.log", vulkan_log=root / "vulkan.log",
                hrx_device="HRX0", vulkan_device="Vulkan0",
                context_lengths=(32, 64), decode_tokens=16, repetitions=1,
                batch_size=32, ubatch_size=16, gpu_layers=99, flash_attn="on",
                run_timeout_seconds=10, batched=False, batch_number=None,
            )

            def fake_measurement(_binary, _model, _device, context, _model_context, kind, _args, _log):
                if context == 32:
                    return {"status": "unsupported", "failure_kind": "model-context",
                            "reason": "model context limit"}
                if kind == "decode" and _device == "HRX0":
                    return {"status": "failed", "failure_kind": "model-execution",
                            "error": "decode failed"}
                if kind == "decode" and _device == "Vulkan0":
                    return {"status": "unsupported", "failure_kind": "infrastructure",
                            "error": "device unavailable"}
                return {"status": "ok", "tokens_per_second": 1.0}

            with patch.object(run_context_sweep, "resolve_models", return_value=[(spec, model_path)]), \
                    patch.object(run_context_sweep, "inspect_model", return_value=(128, None)), \
                    patch.object(run_context_sweep, "run_measurement", side_effect=fake_measurement):
                self.assertEqual(run_context_sweep.run(args), 1)

            document = json.loads(args.output.read_text(encoding="utf-8"))
            model = document["models"][0]
            self.assertEqual(model["contexts"]["32"]["hrx"]["result"], "fail")
            self.assertEqual(model["contexts"]["32"]["hrx"]["outcome"], "XFAIL")
            self.assertEqual(model["contexts"]["64"]["hrx"]["outcome"], "XFAIL")
            self.assertEqual(model["contexts"]["64"]["vulkan"]["outcome"], "FAIL")

    def test_streaming_runner_uses_structured_model_and_infrastructure_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = ModelSpec(
                id="model", tier="all",
                expected_results={"long-context-perplexity": {
                    "hrx": {"64": "fail"}, "vulkan": {"64": "report-only"},
                }},
                test_modes={"long-context-perplexity": ("64",)},
                name="Model", repository="owner/model", revision="rev",
                filename="model.gguf", size_bytes=1, sha256="0" * 64,
            )
            model = run_perplexity_benchmark.ResolvedModel(
                spec=spec, path=root / "model.gguf",
            )
            args = argparse.Namespace(run_timeout_seconds=10, scored_tokens=16)
            cases = (
                ("model-execution", "fail", "XFAIL"),
                ("infrastructure", "report-only", "FAIL"),
            )
            for failure_kind, expectation, expected_outcome in cases:
                output = json.dumps({
                    "status": "failed", "failure_kind": failure_kind,
                    "reason": f"test {failure_kind}",
                })
                with self.subTest(failure_kind=failure_kind), patch.object(
                    run_perplexity_benchmark.subprocess, "run",
                    return_value=SimpleNamespace(returncode=1, stdout=output + "\n"),
                ):
                    measurement = run_perplexity_benchmark._run_stream_measurement(
                        root / "evaluator", model, root / "corpus", "HRX0", 64,
                        args, root / f"{failure_kind}.log",
                    )
                self.assertEqual(measurement["failure_kind"], failure_kind)
                self.assertEqual(
                    classify_outcome(
                        "fail", expectation,
                        expected_failure_allowed=failure_kind == "model-execution",
                        infrastructure_failure=failure_kind == "infrastructure",
                    ),
                    expected_outcome,
                )

    def test_long_ppl_worker_applies_xfail_and_infrastructure_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluator = root / "evaluator"
            corpus = root / "corpus.txt"
            evaluator.touch()
            corpus.write_text("corpus", encoding="utf-8")
            spec = ModelSpec(
                id="model", tier="all",
                expected_results={"long-context-perplexity": {
                    "hrx": {"64": "fail"}, "vulkan": {"64": "report-only"},
                }},
                test_modes={"long-context-perplexity": ("64",)},
                name="Model", repository="owner/model", revision="rev",
                filename="model.gguf", size_bytes=1, sha256="0" * 64,
            )
            model = run_perplexity_benchmark.ResolvedModel(
                spec=spec, path=root / "model.gguf",
            )
            args = argparse.Namespace(
                stream_perplexity=evaluator, corpus_file=corpus,
                models_dir=root, model_manifest=root / "manifest.json", models=["Model"],
                context_lengths=(64,), scored_tokens=16, max_perplexity_ratio=1.1,
                hrx_device="HRX0", vulkan_device="Vulkan0", batch_number=None,
                batched=False, output=root / "out.json", hrx_log=root / "hrx.log",
                vulkan_log=root / "vulkan.log",
            )
            cases = (
                ("model-execution", None, 0, "XFAIL", "REPORT_ONLY", "fail"),
                (None, "infrastructure", 1, "XPASS", "FAIL", "pass"),
            )
            for index, (hrx_failure_kind, vulkan_failure_kind, expected_exit,
                        expected_hrx_outcome, expected_vulkan_outcome, expected_result) in enumerate(cases):
                args.output = root / f"case-{index}.json"
                args.hrx_log = root / f"case-{index}-hrx.log"
                args.vulkan_log = root / f"case-{index}-vulkan.log"

                def fake_stream_measurement(_evaluator, _model, _corpus, device, _context, _args, _log):
                    failure_kind = hrx_failure_kind if device == "HRX0" else vulkan_failure_kind
                    if failure_kind is not None:
                        return {"status": "failed", "failure_kind": failure_kind,
                                "error": "structured failure"}
                    return {"status": "ok", "failure_kind": None,
                            "ppl": {"value": 3.0, "uncertainty": 0.0}}

                with self.subTest(case=index), \
                        patch.object(run_perplexity_benchmark, "resolve_models", return_value=[model]), \
                        patch.object(run_perplexity_benchmark, "_run_stream_measurement", side_effect=fake_stream_measurement):
                    self.assertEqual(
                        run_perplexity_benchmark.run_streaming_perplexity(args), expected_exit,
                    )
                row = json.loads(args.output.read_text(encoding="utf-8"))["models"][0]
                pair = row["contexts"]["64"]
                self.assertEqual(row["result"], expected_result)
                self.assertEqual(pair["hrx_outcome"], expected_hrx_outcome)
                self.assertEqual(pair["vulkan_outcome"], expected_vulkan_outcome)

    def test_context_sweep_runner_separates_model_and_device_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(
                decode_tokens=16, batch_size=32, ubatch_size=16, repetitions=1,
                gpu_layers=99, flash_attn="on", run_timeout_seconds=10,
            )
            cases = (
                ("decode failed", "model-execution", "XFAIL"),
                ("unsupported device: Vulkan0", "infrastructure", "FAIL"),
            )
            for diagnostic, failure_kind, expected_outcome in cases:
                with self.subTest(diagnostic=diagnostic), patch.object(
                    run_context_sweep.subprocess, "run",
                    return_value=SimpleNamespace(returncode=1, stdout=diagnostic, stderr=""),
                ):
                    measurement = run_context_sweep.run_measurement(
                        root / "llama-bench", root / "model.gguf", "Vulkan0", 64, 128,
                        "decode", args, root / f"{failure_kind}.log",
                    )
                self.assertEqual(measurement["failure_kind"], failure_kind)
                self.assertEqual(
                    classify_outcome(
                        "fail", "fail" if failure_kind == "model-execution" else "report-only",
                        expected_failure_allowed=failure_kind == "model-execution",
                        infrastructure_failure=failure_kind == "infrastructure",
                    ),
                    expected_outcome,
                )

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
