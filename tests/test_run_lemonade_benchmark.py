# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPOSITORY_ROOT / "scripts" / "hrx" / "run_lemonade_benchmark.py"
REPORTER_PATH = (
    REPOSITORY_ROOT / "scripts" / "hrx" / "write_lemonade_benchmark_report.py"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = load_module("run_lemonade_benchmark_tested", WORKER_PATH)


def make_model(name: str, backend: str, *, failed_runs: int = 0) -> dict[str, Any]:
    return {
        "model": name,
        "results": [
            {
                "recipe": "llamacpp",
                "backend": backend,
                "ctx_size": 2048,
                "backend_args": "",
                "scenarios": [
                    {
                        "name": "smoke",
                        "failed_runs": failed_runs,
                        "output_tokens": 16,
                        "ttft_ms": {"mean": 10.0, "min": 9.0, "max": 11.0},
                        "tps": {"mean": 20.0, "min": 19.0, "max": 21.0},
                        "vram_peak_gb": 2.0,
                    }
                ],
            }
        ],
    }


def make_benchmark(
    names: list[str],
    backend: str,
    *,
    timestamp: str = "2026-08-15T00:00:00Z",
    hardware: dict[str, Any] | None = None,
    failed_runs: int = 0,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "hardware": hardware or {"gpu": "gfx1151", "backends": [backend]},
        "models": [
            make_model(name, backend, failed_runs=failed_runs) for name in names
        ],
    }


class BenchmarkWorkerTest(unittest.TestCase):
    def make_args(
        self,
        root: Path,
        *,
        models_dir: Path | None = None,
        batched: bool = False,
        batch_number: int | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            lemonade_build_dir=root / "lemonade-build",
            llama_server=root / "llama-server",
            state_root=root / "state",
            models_dir=models_dir,
            batched=batched,
            batch_number=batch_number,
            hrx_output=root / "output" / "hrx.json",
            vulkan_output=root / "output" / "vulkan.json",
            hrx_server_log=root / "logs" / "hrx-server.log",
            vulkan_server_log=root / "logs" / "vulkan-server.log",
            hrx_response_log=root / "logs" / "hrx-responses.jsonl",
            vulkan_response_log=root / "logs" / "vulkan-responses.jsonl",
            models=["extra.model-a", "extra.model-b"],
        )

    def test_prepare_state_replaces_existing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_root = Path(temporary_directory) / "state"
            cache_dir = state_root / "lemonade"
            cache_dir.mkdir(parents=True)
            sentinel = cache_dir / "remove-me.txt"
            sentinel.write_text("old", encoding="utf-8")

            actual_cache, hf_home, runtime_dir = worker.prepare_state(state_root)

            self.assertEqual(actual_cache, cache_dir)
            self.assertFalse(sentinel.exists())
            self.assertTrue(hf_home.is_dir())
            self.assertTrue(runtime_dir.is_dir())
            self.assertEqual(runtime_dir.stat().st_mode & 0o777, 0o700)

    def test_run_benchmark_preserves_overwrite_and_auto_pull_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "benchmark.json"
            output.write_text("stale", encoding="utf-8")
            commands: list[list[str]] = []

            def complete_benchmark(command: list[str], **_: object) -> None:
                commands.append(command)
                output_path = Path(command[command.index("--output") + 1])
                self.assertFalse(output_path.exists())
                output_path.write_text(
                    json.dumps(make_benchmark(["model-a"], "hrx")),
                    encoding="utf-8",
                )

            with mock.patch.object(
                worker.subprocess, "run", side_effect=complete_benchmark
            ):
                result = worker.run_benchmark(
                    root / "lemonade",
                    "hrx",
                    output,
                    root / "responses.jsonl",
                    ["model-a"],
                    auto_pull=True,
                    env={},
                )

            self.assertIsNone(result)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["models"][0][
                    "model"
                ],
                "model-a",
            )
            self.assertIn("--auto-pull", commands[0])

    def test_run_benchmark_rejects_entire_failed_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def complete_benchmark(command: list[str], **_: object) -> None:
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        make_benchmark(
                            ["model-a", "model-b"], "hrx", failed_runs=1
                        )
                    ),
                    encoding="utf-8",
                )

            with mock.patch.object(
                worker.subprocess, "run", side_effect=complete_benchmark
            ):
                with self.assertRaisesRegex(
                    worker.BenchmarkValidationError, "reported 1 failed run"
                ):
                    worker.run_benchmark(
                        root / "lemonade",
                        "hrx",
                        root / "benchmark.json",
                        root / "responses.jsonl",
                        ["model-a", "model-b"],
                        auto_pull=False,
                        env={},
                    )

    def test_local_config_sets_and_verifies_extra_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models_dir = (root / "models").resolve()
            models_dir.mkdir()
            llama_server = (root / "llama-server").resolve()
            config = {
                "llamacpp": {
                    "hrx_bin": str(llama_server),
                    "vulkan_bin": str(llama_server),
                    "device": "HRX0",
                },
                "no_fetch_executables": True,
                "log_level": "debug",
                "extra_models_dir": str(models_dir),
            }

            with (
                mock.patch.object(worker.subprocess, "run") as run_command,
                mock.patch.object(worker, "request_json", return_value=config),
            ):
                worker.set_lemonade_config_values(
                    root / "lemonade",
                    llama_server,
                    "HRX0",
                    models_dir=models_dir,
                    env={},
                    port=13305,
                )

            command = run_command.call_args.args[0]
            self.assertIn(f"extra_models_dir={models_dir}", command)

    def test_config_without_models_dir_remains_branch_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llama_server = (root / "llama-server").resolve()
            config = {
                "llamacpp": {
                    "hrx_bin": str(llama_server),
                    "vulkan_bin": str(llama_server),
                    "device": "HRX0",
                },
                "no_fetch_executables": True,
                "log_level": "debug",
            }

            with (
                mock.patch.object(worker.subprocess, "run") as run_command,
                mock.patch.object(worker, "request_json", return_value=config),
            ):
                worker.set_lemonade_config_values(
                    root / "lemonade",
                    llama_server,
                    "HRX0",
                    env={},
                    port=13305,
                )

            self.assertFalse(
                any(
                    argument.startswith("extra_models_dir=")
                    for argument in run_command.call_args.args[0]
                )
            )

    def test_merge_keeps_first_timestamp_and_appends_models_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "results" / "benchmark.json"
            first = make_benchmark(
                ["model-a"], "hrx", timestamp="2026-08-15T00:00:00Z"
            )
            second = make_benchmark(
                ["model-b"], "hrx", timestamp="2026-08-15T01:00:00Z"
            )

            self.assertEqual(worker.merge_benchmark_output(output, first), 1)
            self.assertEqual(worker.merge_benchmark_output(output, second), 1)

            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(merged["timestamp"], first["timestamp"])
            self.assertEqual(
                [entry["model"] for entry in merged["models"]],
                ["model-a", "model-b"],
            )
            self.assertEqual(list(output.parent.glob(".benchmark.json.*.tmp")), [])

    def test_merge_rejections_leave_cumulative_output_unchanged(self) -> None:
        original = make_benchmark(["model-a"], "hrx")
        cases = (
            (
                "duplicate across batches",
                make_benchmark(["model-a"], "hrx"),
                "Duplicate model",
            ),
            (
                "duplicate within batch",
                {
                    **make_benchmark(["model-b"], "hrx"),
                    "models": [
                        make_model("model-b", "hrx"),
                        make_model("model-b", "hrx"),
                    ],
                },
                "Duplicate model",
            ),
            (
                "hardware drift",
                make_benchmark(
                    ["model-b"],
                    "hrx",
                    hardware={"gpu": "different", "backends": ["hrx"]},
                ),
                "metadata incompatible",
            ),
            (
                "failed scenario",
                make_benchmark(["model-b"], "hrx", failed_runs=1),
                "reported 1 failed run",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "benchmark.json"
            for label, batch, error in cases:
                with self.subTest(label=label):
                    output.write_text(json.dumps(original), encoding="utf-8")
                    before = output.read_bytes()

                    with self.assertRaisesRegex(RuntimeError, error):
                        worker.merge_benchmark_output(output, batch)

                    self.assertEqual(output.read_bytes(), before)

    def test_atomic_publication_failure_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "benchmark.json"
            original = make_benchmark(["model-a"], "hrx")
            output.write_text(json.dumps(original), encoding="utf-8")
            before = output.read_bytes()

            with mock.patch.object(
                worker.os, "replace", side_effect=OSError("publication failed")
            ):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    worker.merge_benchmark_output(
                        output, make_benchmark(["model-b"], "hrx")
                    )

            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(output.parent.glob(".benchmark.json.*.tmp")), [])

    def test_append_batch_log_labels_present_and_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "cumulative.log"
            destination.write_text("earlier\n", encoding="utf-8")
            source = root / "source.log"
            source.write_text("current\n", encoding="utf-8")

            worker.append_batch_log(destination, source, 2)
            worker.append_batch_log(destination, root / "missing.log", 3)

            content = destination.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("earlier\n"))
            self.assertIn("===== BEGIN batch 2 =====\ncurrent\n", content)
            self.assertIn("===== END batch 2 =====", content)
            self.assertIn("===== BEGIN batch 3 =====", content)
            self.assertIn("===== END batch 3 =====", content)

    def test_non_batched_run_keeps_original_paths_flow_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root)
            for added_attribute in ("models_dir", "batched", "batch_number"):
                delattr(args, added_attribute)
            args.hrx_server_log.parent.mkdir(parents=True)
            args.hrx_server_log.write_text("stale", encoding="utf-8")
            args.hrx_response_log.write_text("stale", encoding="utf-8")

            def benchmark_result(
                _executable: Path,
                backend: str,
                _output: Path,
                _response: Path,
                models: list[str],
                **_: object,
            ) -> dict[str, Any]:
                return make_benchmark(models, backend)

            with (
                mock.patch.object(worker, "log"),
                mock.patch.object(worker, "log_llama_server_devices"),
                mock.patch.object(
                    worker,
                    "start_lemond",
                    side_effect=[mock.sentinel.hrx, mock.sentinel.vulkan],
                ) as start_server,
                mock.patch.object(worker, "wait_for_live"),
                mock.patch.object(worker, "set_lemonade_config_values") as configure,
                mock.patch.object(
                    worker, "run_benchmark", side_effect=benchmark_result
                ) as run_benchmark,
                mock.patch.object(worker, "stop_server", return_value=0),
                mock.patch.object(worker, "verify_configured_executable") as verify,
            ):
                result = worker.run(args)

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[2] for call in start_server.call_args_list],
                [args.hrx_server_log, args.vulkan_server_log],
            )
            self.assertEqual(
                [call.args[2] for call in run_benchmark.call_args_list],
                [args.hrx_output, args.vulkan_output],
            )
            self.assertEqual(
                [call.args[3] for call in run_benchmark.call_args_list],
                [args.hrx_response_log, args.vulkan_response_log],
            )
            self.assertEqual(
                [call.kwargs["auto_pull"] for call in run_benchmark.call_args_list],
                [True, False],
            )
            self.assertTrue(
                all(
                    call.kwargs["models_dir"] is None
                    for call in configure.call_args_list
                )
            )
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(args.hrx_server_log.read_text(encoding="utf-8"), "")
            self.assertEqual(args.hrx_response_log.read_text(encoding="utf-8"), "")
            self.assertFalse(args.state_root.exists())

    def test_local_models_disable_auto_pull_for_both_original_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models_dir = root / "models"
            models_dir.mkdir()
            args = self.make_args(root, models_dir=models_dir)

            with (
                mock.patch.object(worker, "log"),
                mock.patch.object(worker, "log_llama_server_devices"),
                mock.patch.object(
                    worker,
                    "start_lemond",
                    side_effect=[mock.sentinel.hrx, mock.sentinel.vulkan],
                ),
                mock.patch.object(worker, "wait_for_live"),
                mock.patch.object(worker, "set_lemonade_config_values") as configure,
                mock.patch.object(
                    worker,
                    "run_benchmark",
                    side_effect=lambda _, backend, _o, _r, models, **_kw: (
                        make_benchmark(models, backend)
                    ),
                ) as run_benchmark,
                mock.patch.object(worker, "stop_server", return_value=0),
                mock.patch.object(worker, "verify_configured_executable"),
            ):
                worker.run(args)

            self.assertTrue(
                all(
                    call.kwargs["models_dir"] == models_dir.resolve()
                    for call in configure.call_args_list
                )
            )
            self.assertTrue(
                all(
                    call.kwargs["auto_pull"] is False
                    for call in run_benchmark.call_args_list
                )
            )

    def test_batched_run_uses_temporaries_merges_after_verification_and_appends_logs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models_dir = root / "models"
            models_dir.mkdir()
            args = self.make_args(
                root, models_dir=models_dir, batched=True, batch_number=2
            )
            for path in (
                args.hrx_server_log,
                args.vulkan_server_log,
                args.hrx_response_log,
                args.vulkan_response_log,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("batch 1 content\n", encoding="utf-8")

            events: list[tuple[str, str]] = []
            temporary_paths: list[Path] = []

            def start_server(
                _build: Path,
                _cache: Path,
                log_path: Path,
                env: dict[str, str],
                port: int,
            ) -> object:
                del env, port
                temporary_paths.append(log_path)
                backend = "hrx" if "hrx" in log_path.parent.name else "vulkan"
                log_path.write_text(f"{backend} server\n", encoding="utf-8")
                return object()

            def complete_benchmark(
                _executable: Path,
                backend: str,
                output: Path,
                response_log: Path,
                models: list[str],
                **_: object,
            ) -> dict[str, Any]:
                temporary_paths.extend((output, response_log))
                response_log.write_text(f"{backend} response\n", encoding="utf-8")
                data = make_benchmark(models, backend, timestamp=f"batch-2-{backend}")
                output.write_text(json.dumps(data), encoding="utf-8")
                return data

            def stop_server(_process: object) -> int:
                events.append(("stop", "phase"))
                return 0

            def verify(**kwargs: object) -> None:
                phases = kwargs["phases"]
                index = kwargs["phase_index"]
                backend = phases[index].backend
                self.assertTrue(phases[index].server_log.is_file())
                events.append(("verify", backend))

            actual_merge = worker.merge_benchmark_output

            def merge(output: Path, data: dict[str, Any]) -> int:
                backend = data["models"][0]["results"][0]["backend"]
                events.append(("merge", backend))
                return actual_merge(output, data)

            with (
                mock.patch.object(worker, "log"),
                mock.patch.object(worker, "log_llama_server_devices"),
                mock.patch.object(worker, "start_lemond", side_effect=start_server),
                mock.patch.object(worker, "wait_for_live"),
                mock.patch.object(worker, "set_lemonade_config_values"),
                mock.patch.object(
                    worker, "run_benchmark", side_effect=complete_benchmark
                ) as run_benchmark,
                mock.patch.object(worker, "stop_server", side_effect=stop_server),
                mock.patch.object(
                    worker, "verify_configured_executable", side_effect=verify
                ),
                mock.patch.object(
                    worker, "merge_benchmark_output", side_effect=merge
                ),
            ):
                result = worker.run(args)

            self.assertEqual(result, 0)
            self.assertEqual(
                events,
                [
                    ("stop", "phase"),
                    ("verify", "hrx"),
                    ("merge", "hrx"),
                    ("stop", "phase"),
                    ("verify", "vulkan"),
                    ("merge", "vulkan"),
                ],
            )
            self.assertTrue(
                all(
                    call.kwargs["auto_pull"] is False
                    for call in run_benchmark.call_args_list
                )
            )
            self.assertTrue(all(not path.exists() for path in temporary_paths))
            for output in (args.hrx_output, args.vulkan_output):
                self.assertEqual(
                    [entry["model"] for entry in json.loads(
                        output.read_text(encoding="utf-8")
                    )["models"]],
                    args.models,
                )
            for path in (
                args.hrx_server_log,
                args.vulkan_server_log,
                args.hrx_response_log,
                args.vulkan_response_log,
            ):
                content = path.read_text(encoding="utf-8")
                self.assertTrue(content.startswith("batch 1 content\n"))
                self.assertIn("===== BEGIN batch 2 =====", content)
                self.assertIn("===== END batch 2 =====", content)
            self.assertFalse(args.state_root.exists())

    def test_batched_failure_appends_logs_but_does_not_merge_or_run_vulkan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models_dir = root / "models"
            models_dir.mkdir()
            args = self.make_args(
                root, models_dir=models_dir, batched=True, batch_number=2
            )
            existing = make_benchmark(["existing"], "hrx")
            args.hrx_output.parent.mkdir(parents=True)
            args.hrx_output.write_text(json.dumps(existing), encoding="utf-8")
            before = args.hrx_output.read_bytes()

            def start_server(
                _build: Path,
                _cache: Path,
                log_path: Path,
                env: dict[str, str],
                port: int,
            ) -> object:
                del env, port
                log_path.write_text("server failure details\n", encoding="utf-8")
                return object()

            def fail_benchmark(
                _executable: Path,
                _backend: str,
                _output: Path,
                response_log: Path,
                _models: list[str],
                **_: object,
            ) -> dict[str, Any]:
                response_log.write_text("response failure details\n", encoding="utf-8")
                raise worker.BenchmarkValidationError("invalid batch")

            with (
                mock.patch.object(worker, "log"),
                mock.patch.object(worker, "log_llama_server_devices"),
                mock.patch.object(
                    worker, "start_lemond", side_effect=start_server
                ) as start,
                mock.patch.object(worker, "wait_for_live"),
                mock.patch.object(worker, "set_lemonade_config_values"),
                mock.patch.object(worker, "run_benchmark", side_effect=fail_benchmark),
                mock.patch.object(worker, "stop_server", return_value=0),
                mock.patch.object(worker, "verify_configured_executable") as verify,
                mock.patch.object(worker, "merge_benchmark_output") as merge,
            ):
                with self.assertRaisesRegex(
                    worker.BenchmarkValidationError, "invalid batch"
                ):
                    worker.run(args)

            self.assertEqual(start.call_count, 1)
            verify.assert_not_called()
            merge.assert_not_called()
            self.assertEqual(args.hrx_output.read_bytes(), before)
            self.assertIn(
                "server failure details",
                args.hrx_server_log.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "response failure details",
                args.hrx_response_log.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "===== BEGIN batch 2 =====",
                args.vulkan_server_log.read_text(encoding="utf-8"),
            )
            self.assertFalse(args.state_root.exists())

    def test_executable_verification_failure_contributes_no_batched_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root, batched=True, batch_number=1)

            def start_server(
                _build: Path,
                _cache: Path,
                log_path: Path,
                env: dict[str, str],
                port: int,
            ) -> object:
                del env, port
                log_path.write_text("server\n", encoding="utf-8")
                return object()

            def benchmark(
                _executable: Path,
                backend: str,
                output: Path,
                response_log: Path,
                models: list[str],
                **_: object,
            ) -> None:
                response_log.write_text("response\n", encoding="utf-8")
                output.write_text(
                    json.dumps(make_benchmark(models, backend)), encoding="utf-8"
                )

            with (
                mock.patch.object(worker, "log"),
                mock.patch.object(worker, "log_llama_server_devices"),
                mock.patch.object(worker, "start_lemond", side_effect=start_server),
                mock.patch.object(worker, "wait_for_live"),
                mock.patch.object(worker, "set_lemonade_config_values"),
                mock.patch.object(worker, "run_benchmark", side_effect=benchmark),
                mock.patch.object(worker, "stop_server", return_value=0),
                mock.patch.object(
                    worker,
                    "verify_configured_executable",
                    side_effect=RuntimeError("wrong executable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "wrong executable"):
                    worker.run(args)

            self.assertFalse(args.hrx_output.exists())
            self.assertFalse(args.vulkan_output.exists())
            self.assertIn(
                "===== BEGIN batch 1 =====",
                args.hrx_server_log.read_text(encoding="utf-8"),
            )

    def test_cli_accepts_generic_batcher_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            argv = [
                "run_lemonade_benchmark.py",
                "--lemonade-build-dir",
                str(root / "build"),
                "--llama-server",
                str(root / "server"),
                "--state-root",
                str(root / "state"),
                "--hrx-output",
                str(root / "hrx.json"),
                "--vulkan-output",
                str(root / "vulkan.json"),
                "--hrx-server-log",
                str(root / "hrx.log"),
                "--vulkan-server-log",
                str(root / "vulkan.log"),
                "--hrx-response-log",
                str(root / "hrx.jsonl"),
                "--vulkan-response-log",
                str(root / "vulkan.jsonl"),
                "--batched",
                "--batch-number",
                "3",
                "--models-dir",
                str(root / "models"),
                "--models",
                "extra.model-a",
                "extra.model-b",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(worker, "run", return_value=0) as run,
            ):
                self.assertEqual(worker.main(), 0)

            args = run.call_args.args[0]
            self.assertTrue(args.batched)
            self.assertEqual(args.batch_number, 3)
            self.assertEqual(args.models_dir, root / "models")
            self.assertEqual(args.models, ["extra.model-a", "extra.model-b"])

    def test_cli_requires_positive_batch_number_only_with_batched(self) -> None:
        required = [
            "run_lemonade_benchmark.py",
            "--lemonade-build-dir",
            "build",
            "--llama-server",
            "server",
            "--state-root",
            "state",
            "--hrx-output",
            "hrx.json",
            "--vulkan-output",
            "vulkan.json",
            "--hrx-server-log",
            "hrx.log",
            "--vulkan-server-log",
            "vulkan.log",
            "--hrx-response-log",
            "hrx.jsonl",
            "--vulkan-response-log",
            "vulkan.jsonl",
            "--models",
            "model-a",
        ]
        invalid_suffixes = (
            ["--batched"],
            ["--batch-number", "1"],
            ["--batched", "--batch-number", "0"],
        )
        for suffix in invalid_suffixes:
            with self.subTest(suffix=suffix):
                with mock.patch.object(sys, "argv", [*required, *suffix]):
                    with self.assertRaises(SystemExit):
                        worker.main()

    def test_branch_original_reporter_consumes_multi_batch_results(self) -> None:
        reporter = load_module("lemonade_reporter_integration_tested", REPORTER_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hrx_output = root / "hrx.json"
            vulkan_output = root / "vulkan.json"
            for backend, output in (
                ("hrx", hrx_output),
                ("vulkan", vulkan_output),
            ):
                worker.merge_benchmark_output(
                    output,
                    make_benchmark(
                        ["model-a"], backend, timestamp="first timestamp"
                    ),
                )
                worker.merge_benchmark_output(
                    output,
                    make_benchmark(
                        ["model-b"], backend, timestamp="second timestamp"
                    ),
                )

            report = reporter.format_report(
                reporter.load_benchmark(hrx_output),
                reporter.load_benchmark(vulkan_output),
            )

            self.assertIn("`model-a`", report)
            self.assertIn("`model-b`", report)
            self.assertIn("Lemonade HRX/Vulkan comparison", report)


if __name__ == "__main__":
    unittest.main()
