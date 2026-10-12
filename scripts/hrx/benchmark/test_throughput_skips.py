#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise throughput skips without a GPU or a running Lemonade daemon."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmark_output import merge_benchmark_output
from benchmark_report import ReportError
import run_lemonade_benchmark as worker
import run_perplexity_benchmark as perplexity
from write_lemonade_benchmark_report import format_report


def measurement(backend, models, failed=False):
    return {
        "timestamp": "2026-10-12T00:00:00Z",
        "hardware": {"cpu": "test"},
        "models": [
            {
                "model": model,
                "results": [{
                    "backend": backend,
                    "recipe": "llamacpp",
                    "ctx_size": 4096,
                    "scenarios": [{
                        "name": "chat-short",
                        "failed_runs": 3 if failed else 0,
                        "all_runs_failed": failed,
                        "tps": {"mean": 42.0},
                    }],
                }],
            }
            for model in models
        ],
    }


class ThroughputSkipTests(unittest.TestCase):
    def run_worker(self, root, models, *, vulkan_fails=False):
        models_dir = root / "models"
        models_dir.mkdir(exist_ok=True)
        args = Namespace(
            lemonade_build_dir=root / "lemonade",
            llama_server=root / "llama-server",
            models_dir=models_dir,
            hrx_xfail_models=[],
            hrx_report_only_models=[],
            hrx_skip_models=["bad"],
            models=models,
            batched=True,
            batch_number=1,
        )
        for backend in ("hrx", "vulkan"):
            for field in ("output", "server_log", "response_log"):
                setattr(args, f"{backend}_{field}", root / f"{backend}-{field}")

        def benchmark(executable, backend, output, response_log, selected, xfail, **kwargs):
            is_vulkan = backend == "vulkan"
            fail_measurement = is_vulkan and vulkan_fails
            data = measurement(backend, selected, fail_measurement)
            _, _, unexpected = worker.summarize_benchmark(data, backend, selected, xfail)
            return data, unexpected

        runner = Mock(side_effect=benchmark)
        with patch.multiple(
            worker,
            log=Mock(),
            log_llama_server_devices=Mock(),
            start_lemond=Mock(),
            stop_server=Mock(return_value=0),
            wait_for_live=Mock(),
            set_lemonade_config_values=Mock(),
            verify_configured_executable=Mock(),
            run_benchmark=runner,
        ):
            status = worker.run(args)
        documents = {
            backend: json.loads(getattr(args, f"{backend}_output").read_text())
            for backend in ("hrx", "vulkan")
        }
        calls = [(call.args[1], call.args[4]) for call in runner.call_args_list]
        return status, documents, calls

    def test_skip_omits_hrx_execution_but_keeps_vulkan_and_visible_result(self):
        with tempfile.TemporaryDirectory() as directory:
            status, documents, calls = self.run_worker(Path(directory), ["bad", "good"])
        self.assertEqual(status, 0)
        self.assertEqual(calls, [("hrx", ["good"]), ("vulkan", ["bad", "good"])])
        rows = {row["model"]: row for row in documents["hrx"]["models"]}
        self.assertEqual(rows["bad"]["results"], [])
        self.assertEqual(rows["bad"]["outcome"], "SKIP")
        report = format_report(documents["hrx"], documents["vulkan"])
        self.assertIn("| bad | 42.0 | — | — | SKIP |", report)
        self.assertIn("| good | 42.0 | 42.0 | 100.0% | PASS |", report)

        missing = deepcopy(documents["hrx"])
        missing["models"][0]["results"] = []
        with self.assertRaises(ReportError):
            format_report(missing, documents["vulkan"])
        missing = deepcopy(documents["vulkan"])
        missing["models"] = missing["models"][1:]
        with self.assertRaises(ReportError):
            format_report(documents["hrx"], missing)

    def test_all_skipped_batch_still_enforces_vulkan_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status, documents, calls = self.run_worker(root, ["bad"], vulkan_fails=True)
            self.assertEqual(status, 1)
            self.assertEqual(calls, [("vulkan", ["bad"])])
            self.assertEqual(documents["hrx"]["hardware"], {})
            self.assertEqual(documents["vulkan"]["models"][0]["outcome"], "FAIL")
            self.assertIn("SKIP", format_report(documents["hrx"], documents["vulkan"]))

            merge_benchmark_output(root / "hrx-output", measurement("hrx", ["good"]))
            merged = json.loads((root / "hrx-output").read_text())
            self.assertEqual(merged["hardware"], {"cpu": "test"})
            self.assertEqual([row["model"] for row in merged["models"]], ["bad", "good"])

    def test_merge_without_hardware_remains_valid_for_perplexity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "perplexity.json"
            merge_benchmark_output(output, {"models": [{"model": "one"}]})
            merge_benchmark_output(output, {"models": [{"model": "two"}]})
            self.assertEqual(json.loads(output.read_text()), {
                "models": [{"model": "one"}, {"model": "two"}],
            })

    def test_perplexity_recovery_still_requires_removing_expected_failure(self):
        document = {
            "settings": {"max_perplexity_ratio": 1.1},
            "models": [{
                "model": "bad",
                "regimes": {"decode-like": {
                    "hrx": {"status": "ok", "ppl": {"value": 30.0}},
                    "vulkan": {"status": "ok", "ppl": {"value": 10.0}},
                }},
            }],
        }
        with patch.object(perplexity, "log"):
            self.assertFalse(perplexity.evaluate_models(document, {"bad"}, set()))
            self.assertEqual(document["models"][0]["outcome"], "XFAIL")
            document["models"][0]["regimes"]["decode-like"]["hrx"]["ppl"]["value"] = 10.0
            self.assertTrue(perplexity.evaluate_models(document, {"bad"}, set()))
            self.assertEqual(document["models"][0]["outcome"], "XPASS")
            self.assertFalse(perplexity.evaluate_models(document, set(), set()))
            self.assertEqual(document["models"][0]["outcome"], "PASS")


if __name__ == "__main__":
    unittest.main()
