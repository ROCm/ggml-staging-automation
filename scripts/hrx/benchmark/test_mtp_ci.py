# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for MTP case isolation, response capture and matched reports."""
import copy
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import gzip
import http.client
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lemonade_capture import capture_responses, draft_metrics
from run_batched_benchmark import BatchBenchmarkError, load_manifest
from run_lemonade_benchmark import classify, register_models, run_case
from run_perplexity_benchmark import evaluate_models
from write_lemonade_benchmark_report import format_report
from benchmark_report import ReportError

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hrx/package"))
from fetch_lemonade import install


class MtpCI(unittest.TestCase):
    def test_release_checksum_failure_preserves_existing_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.tar.gz"
            archive.write_bytes(b"wrong bytes")
            pin = root / "pin.json"
            pin.write_text(json.dumps({"tag": "v1", "asset": "archive.tar.gz", "sha256": "0" * 64}))
            output = root / "install"
            output.mkdir()
            marker = output / "existing"
            marker.write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                install(pin, output, archive)
            self.assertEqual(marker.read_text(), "keep")

    def test_full_manifest_and_independent_expectations(self):
        manifest = load_manifest(ROOT / "benchmarks/hrx/model_manifest.json")
        self.assertGreater(len(manifest.models), 1)
        qwen = next(m for m in manifest.models if m.id == "qwen3.5-35b-a3b-mtp")
        self.assertEqual(qwen.mtp, ("off", "on"))
        self.assertEqual(qwen.expectation("lemonade-benchmark", "hrx", "on"), "fail")
        self.assertEqual(qwen.expectation("lemonade-benchmark", "vulkan", "on"), "pass")
        self.assertEqual(qwen.expectation("lemonade-benchmark", "vulkan", "off"), "pass")
        self.assertEqual(qwen.expectation("perplexity", "hrx"), "fail")
        self.assertTrue(all(not m.name.startswith(("extra.", "local.", "user.")) for m in manifest.models))

    def test_stale_mode_exception_is_rejected(self):
        data = json.loads((ROOT / "benchmarks/hrx/model_manifest.json").read_text())
        m = data["models"][0]
        m["mtp"] = ["off", "on"]
        m["expected_results"] = {"lemonade-benchmark": {"hrx": {"default": "fail"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(BatchBenchmarkError):
                load_manifest(path)

    def test_registration_reuses_same_weights_without_forcing_mtp(self):
        manifest = load_manifest(ROOT / "benchmarks/hrx/model_manifest.json")
        model = next(m for m in manifest.models if m.mtp == ("off", "on"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / model.directory / model.filename
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fixture")
            cache = root / "cache"
            cache.mkdir()
            register_models([model], root / "models", cache, root / "hf")
            registrations = json.loads((cache / "user_models.json").read_text())
            self.assertEqual(len(registrations), 4)
            self.assertEqual(len({d["checkpoint"] for d in registrations.values()}), 1)
            self.assertEqual({d["recipe"] for d in registrations.values()}, {"llamacpp", "llamacpp-hrx"})
            options = json.loads((cache / "recipe_options.json").read_text())
            for name, value in options.items():
                self.assertTrue(value["merge_args"])
                args = value.get("hrx_args", value.get("llamacpp_args"))
                self.assertEqual("--spec-type none" in args, name.endswith("-off"))
                self.assertNotIn("draft-mtp", args)
                self.assertNotIn("--device", args)
                if "-vulkan-" in name:
                    self.assertEqual(value["llamacpp_device"], "Vulkan0")
            links = list((root / "hf").rglob("*.gguf"))
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0].resolve(), weights)

    def test_counters_and_outcomes(self):
        metrics = draft_metrics([{"status": 200, "response": {"timings": {
            "draft_n": 10, "draft_n_accepted": 0}}}])
        self.assertEqual(metrics["drafted"], 10)
        self.assertEqual(metrics["acceptance_rate"], 0)
        self.assertEqual(classify("fail", False), "XFAIL")
        self.assertEqual(classify("fail", True), "XPASS")
        self.assertEqual(classify("pass", False), "FAIL")
        with self.assertRaises(RuntimeError):
            draft_metrics([{"status": 200, "response": {"timings": {
                "draft_n": 1, "draft_n_accepted": 2}}}])

    def test_on_requires_drafts_but_not_acceptance(self):
        manifest = load_manifest(ROOT / "benchmarks/hrx/model_manifest.json")
        model = replace(manifest.models[0], expected_results={}, mtp=("off", "on"))
        @contextmanager
        def captured(_port, _path):
            yield 1234, records
        def bench(command, **_kwargs):
            name = command[-1]
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({"models": [{"model": name, "results": [{
                "backend": "hrx", "recipe": "llamacpp-hrx", "ctx_size": 4096,
                "scenarios": [{"name": scenario, "failed_runs": 0, "tps": {"mean": 10}}
                              for scenario in ("chat-short", "chat-long-output")]}]}]}))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode, proposed, expected in (("on", 0, "FAIL"), ("on", 4, "PASS"),
                                             ("off", 4, "FAIL"), ("off", 0, "PASS")):
                records = [{"status": 200, "response": {"timings": {
                    "draft_n": proposed, "draft_n_accepted": 0}}}]
                with patch("run_lemonade_benchmark.capture_responses", captured), \
                     patch("run_lemonade_benchmark.subprocess.run", bench):
                    row, _, failed = run_case(Path("lemonade"), model, "hrx", mode,
                                              root, {}, 1234, root / "responses")
                self.assertEqual(row["outcome"], expected)
                self.assertEqual(failed, expected == "FAIL")

    def test_cli_failure_without_artifact_cannot_be_xfailed(self):
        manifest = load_manifest(ROOT / "benchmarks/hrx/model_manifest.json")
        model = next(m for m in manifest.models if m.id == "qwen3.5-35b-a3b-mtp")
        @contextmanager
        def captured(_port, _path):
            yield 1234, []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("run_lemonade_benchmark.capture_responses", captured), \
                 patch("run_lemonade_benchmark.subprocess.run", return_value=SimpleNamespace(returncode=2)):
                with self.assertRaisesRegex(RuntimeError, "without a benchmark artifact"):
                    run_case(Path("lemonade"), model, "hrx", "on", root, {}, 1234, root / "responses")

    def test_perplexity_backend_exceptions_are_independent(self):
        def measurement(value):
            return {"status": "ok", "ppl": {"value": value}}
        row = {"model": "fixture", "expected_results": {"hrx": "fail", "vulkan": "pass"},
               "regimes": {"decode-like": {"hrx": measurement(2), "vulkan": measurement(1)}}}
        document = {"settings": {"max_perplexity_ratio": 1.1}, "models": [row]}
        self.assertFalse(evaluate_models(document))
        self.assertEqual(row["outcome"], "XFAIL")
        self.assertEqual(row["reference_outcome"], "PASS")
        row["regimes"]["decode-like"]["vulkan"] = {"status": "error", "error": "fixture failure"}
        self.assertTrue(evaluate_models(document))
        self.assertEqual(row["reference_outcome"], "FAIL")

    def test_gzip_relay_preserves_body_and_records_counters(self):
        response = {"timings": {"draft_n": 7, "draft_n_accepted": 3}}
        payload = gzip.compress(json.dumps(response).encode())
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with capture_responses(server.server_port, Path(tmp) / "responses") as (port, records):
                    connection = http.client.HTTPConnection("127.0.0.1", port)
                    connection.request("POST", "/api/v1/chat/completions", json.dumps({"model": "fixture"}))
                    reply = connection.getresponse()
                    self.assertEqual(reply.status, 200)
                    self.assertEqual(reply.read(), payload)
                    self.assertEqual(reply.getheader("Content-Encoding"), "gzip")
                    connection.close()
                self.assertEqual(draft_metrics(records)["accepted"], 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_report_matches_modes_not_recipes(self):
        row = {"model": "Example [MTP on]", "outcome": "PASS", "result": "pass", "errors": [],
               "speculative": {"drafted": 10, "accepted": 5, "acceptance_rate": .5},
               "results": [{"recipe": "llamacpp-hrx", "ctx_size": 4096,
                            "scenarios": [{"name": "chat-short", "failed_runs": 0, "tps": {"mean": 100}}]}]}
        hrx = {"models": [row]}
        vulkan = copy.deepcopy(hrx)
        vulkan["models"][0]["results"][0]["recipe"] = "llamacpp"
        self.assertIn("100.0%", format_report(hrx, vulkan))
        hrx["models"][0]["result"] = "fail"
        hrx["models"][0]["outcome"] = "XFAIL"
        self.assertNotIn("100.0%", format_report(hrx, vulkan))
        vulkan["models"][0]["model"] = "Example [MTP off]"
        with self.assertRaises(ReportError):
            format_report(hrx, vulkan)


if __name__ == "__main__":
    unittest.main()
