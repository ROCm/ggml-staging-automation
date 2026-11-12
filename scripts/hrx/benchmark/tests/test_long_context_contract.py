#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Contract checks for long-context benchmark modes and outcomes."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_outcomes import aggregate_status, classify_outcome
import run_context_sweep
import run_perplexity_benchmark
from run_batched_benchmark import ModelSpec, load_manifest


class LongContextOutcomeTests(unittest.TestCase):
    def test_unsupported_coverage_fails_and_only_declared_model_failures_xfail(self) -> None:
        self.assertEqual(aggregate_status(["unsupported"]), "fail")
        self.assertEqual(classify_outcome("fail", "fail"), "XFAIL")
        self.assertEqual(
            classify_outcome("fail", "fail", infrastructure_failure=True), "FAIL"
        )

    def test_manifest_accepts_context_specific_sweep_modes(self) -> None:
        manifest_path = Path(__file__).resolve().parents[4] / "benchmarks/hrx/model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["models"][0]["test-modes"] = {"context-sweep": ["65536"]}
        manifest["models"][0]["expected_results"] = {
            "context-sweep": {"hrx": {"65536": "fail"}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "manifest.json"
            fixture.write_text(json.dumps(manifest), encoding="utf-8")
            resolved = load_manifest(fixture).models[0]
        self.assertEqual(resolved.expectation("context-sweep", "hrx", "65536"), "fail")
        self.assertEqual(resolved.expectation("context-sweep", "hrx", "32768"), "pass")

    def test_context_reader_extracts_architecture_metadata_from_gguf(self) -> None:
        def gguf_string(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.gguf"
            model.write_bytes(
                b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
                + gguf_string("general.architecture") + struct.pack("<I", 8)
                + gguf_string("llama")
                + gguf_string("llama.context_length") + struct.pack("<II", 4, 32768)
            )
            self.assertEqual(run_context_sweep.inspect_model(model), (32768, None))

    def test_long_perplexity_uses_8192_context_and_its_own_xfail_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = ModelSpec(
                id="model", tier="all",
                expected_results={"long-context-perplexity": {
                    "hrx": {"default": "fail"}, "vulkan": {"default": "pass"},
                }},
                test_modes={"long-context-perplexity": ("default",)},
                name="Model", repository="owner/model", revision="rev",
                filename="model.gguf", size_bytes=1, sha256="0" * 64,
            )
            model_path = root / "Model" / "model.gguf"
            model_path.parent.mkdir()
            model_path.touch()
            corpus = root / "corpus.txt"
            corpus.write_text("corpus", encoding="utf-8")
            args = argparse.Namespace(
                llama_perplexity=root / "llama-perplexity", long_context=True,
                models_dir=root, corpus_file=corpus, model_manifest=root / "manifest.json",
                models=["Model"], hrx_device="HRX0", vulkan_device="Vulkan0",
                hrx_log=root / "hrx.log", vulkan_log=root / "vulkan.log",
                perplexity_arg=[], max_perplexity_ratio=1.1, run_timeout_seconds=10,
                batched=False, batch_number=None, output=root / "result.json",
            )
            phases = []

            def fake_phase(phase, _active_log, _models, _binary, _corpus, _args):
                phases.append(phase)
                measurement = {"status": "failed", "error": "execution failed"} if phase.backend == "hrx" else {
                    "status": "ok", "ppl": {"value": 3.0, "uncertainty": 0.0},
                }
                return {"Model": measurement}

            with patch.object(run_perplexity_benchmark, "resolve_models", return_value=[
                run_perplexity_benchmark.ResolvedModel(spec, model_path),
            ]), patch.object(run_perplexity_benchmark, "log_llama_perplexity_devices"), \
                    patch.object(run_perplexity_benchmark, "run_phase", side_effect=fake_phase):
                self.assertEqual(run_perplexity_benchmark.run(args), 0)

            self.assertEqual([phase.regime_id for phase in phases], ["long-context"] * 2)
            document = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(document["regimes"]["long-context"]["ctx"], 8192)
            row = document["models"][0]
            self.assertEqual(row["expected_results"]["hrx"], "fail")
            self.assertEqual(row["outcome"], "XFAIL")

    def test_llama_perplexity_invocation_passes_the_long_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = run_perplexity_benchmark.ResolvedModel(
                ModelSpec(
                    id="model", tier="all", expected_results={}, test_modes={},
                    name="Model", repository="owner/model", revision="rev",
                    filename="model.gguf", size_bytes=1, sha256="0" * 64,
                ), root / "model.gguf",
            )
            args = argparse.Namespace(
                regimes=run_perplexity_benchmark.LONG_CONTEXT_REGIMES,
                perplexity_arg=[], run_timeout_seconds=10,
            )
            phase = run_perplexity_benchmark.PerplexityPhase(
                "HRX long context", "hrx", "HRX0", "long-context", root / "hrx.log",
            )
            with patch.object(
                run_perplexity_benchmark.subprocess, "run",
                return_value=SimpleNamespace(returncode=0, stdout="Final estimate: PPL = 3.0 +/- 0.1\n"),
            ) as invoke, (root / "log.txt").open("w", encoding="utf-8") as log_file:
                measurement, succeeded = run_perplexity_benchmark.run_perplexity(
                    root / "llama-perplexity", model, root / "corpus", phase, args, log_file,
                )
            self.assertTrue(succeeded)
            self.assertEqual(measurement["ppl"]["value"], 3.0)
            command = invoke.call_args.args[0]
            self.assertEqual(command[command.index("-c") + 1], "8192")


if __name__ == "__main__":
    unittest.main()
