#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise video selection, projector staging, and the HTTP worker contract."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest import mock

from run_batched_benchmark import (
    ProjectorSpec,
    download_model,
    load_benchmark_spec,
    load_manifest,
    plan_batches,
    select_models,
)
from run_video_benchmark import run
from write_video_benchmark_report import format_report

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "benchmarks/hrx/model_manifest.json"
SPEC = REPO_ROOT / "benchmarks/hrx/benchmark_spec.json"
TEST_SET = REPO_ROOT / "benchmarks/hrx/video/test_set.json"

FAKE_SERVER = """#!/usr/bin/env python3
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
device = sys.argv[sys.argv.index("--device") + 1]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.answer({"status": "ok"})

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        parts = body["messages"][0]["content"]
        question = parts[0]["text"]
        assert parts[1]["type"] in {"input_video", "image_url"}
        choice = "A" if "Which color" in question else "B"
        if device == "HRX0" and os.environ.get("FAKE_VIDEO_WRONG_HRX") == "1":
            choice = "C"
        self.answer({"choices": [{"message": {"content": choice}}],
            "timings": {"prompt_ms": 12.0, "predicted_ms": 3.0,
                        "predicted_per_second": 42.0}})

    def answer(self, value):
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


class Response(io.BytesIO):
    status = 200


class VideoBenchmarkTest(unittest.TestCase):
    def test_exact_model_selection_and_projector_budget(self) -> None:
        manifest = load_manifest(MANIFEST)
        benchmarks = load_benchmark_spec(SPEC)
        selected = select_models(manifest, "single", '["qwen3.8-27b"]')
        self.assertEqual([model.id for model in selected], ["qwen3.8-27b"])
        video = next(benchmark for benchmark in benchmarks if benchmark.id == "video")
        self.assertEqual(video.required_capabilities, {"video"})
        self.assertTrue(video.required_capabilities <= selected[0].capabilities)
        self.assertEqual(
            selected[0].resident_size_bytes,
            selected[0].size_bytes + selected[0].projector.size_bytes,
        )
        self.assertEqual(
            plan_batches(selected, selected[0].resident_size_bytes), [selected]
        )

    def test_downloads_both_pinned_files(self) -> None:
        model = next(
            model
            for model in load_manifest(MANIFEST).models
            if model.id == "qwen3.8-27b"
        )
        text = b"text"
        projector = b"proj"
        small = replace(
            model,
            size_bytes=len(text),
            sha256=hashlib.sha256(text).hexdigest(),
            projector=ProjectorSpec(
                model.projector.filename,
                len(projector),
                hashlib.sha256(projector).hexdigest(),
            ),
        )
        requested = []

        def urlopen(request, timeout):
            requested.append(request.full_url)
            payload = projector if "mmproj" in request.full_url else text
            return Response(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download_model(small, root, urlopen=urlopen, attempts=1, retry_delay=0)
            self.assertEqual((root / small.name / small.filename).read_bytes(), text)
            self.assertEqual(
                (root / small.name / small.projector.filename).read_bytes(), projector
            )
        self.assertEqual(len(requested), 2)

    def test_server_requests_and_all_answer_gate(self) -> None:
        model = next(
            model
            for model in load_manifest(MANIFEST).models
            if model.id == "qwen3.8-27b"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-llama-server"
            executable.write_text(FAKE_SERVER, encoding="utf-8")
            executable.chmod(0o755)
            args = Namespace(
                llama_server=executable,
                model_manifest=MANIFEST,
                test_set=TEST_SET,
                models_dir=root,
                models=[model.name],
                output=root / "video.json",
                hrx_log=root / "hrx.log",
                vulkan_log=root / "vulkan.log",
                batched=False,
                batch_number=None,
            )
            self.assertEqual(run(args), 0)
            document = json.loads(args.output.read_text())
            self.assertTrue(document["models"][0]["passed"])
            self.assertIn("Vulkan tok/s", format_report(document))
            for backend in ("hrx", "vulkan"):
                cases = document["models"][0]["backends"][backend]["cases"]
                self.assertEqual(len(cases), 3)
                self.assertTrue(all(case["passed"] for case in cases))
                self.assertTrue(all(len(case["samples"]) == 4 for case in cases))
                self.assertEqual(cases[0]["samples"][1]["predicted_tps"], 42.0)

            with mock.patch.dict(os.environ, {"FAKE_VIDEO_WRONG_HRX": "1"}):
                self.assertEqual(run(args), 1)
            failed = json.loads(args.output.read_text())["models"][0]
            self.assertFalse(failed["backends"]["hrx"]["passed"])
            self.assertTrue(failed["backends"]["vulkan"]["passed"])


if __name__ == "__main__":
    unittest.main()
