# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BATCHER_PATH = REPOSITORY_ROOT / "scripts" / "hrx" / "run_batched_benchmark.py"
MANIFEST_PATH = REPOSITORY_ROOT / "scripts" / "hrx" / "lemonade_model_manifest.json"

spec = importlib.util.spec_from_file_location("run_batched_benchmark_tested", BATCHER_PATH)
assert spec is not None and spec.loader is not None
batcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batcher
spec.loader.exec_module(batcher)


def make_manifest_entry(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "model-a",
        "name": "runtime name is opaque",
        "directory": "Model-A",
        "repository": "owner/model-a",
        "revision": "a" * 40,
        "filename": "Model-A.gguf",
        "size_bytes": 7,
        "sha256": hashlib.sha256(b"payload").hexdigest(),
    }
    entry.update(changes)
    return entry


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "models": entries}),
        encoding="utf-8",
    )


def write_benchmark_spec(path: Path, argv: list[object] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmarks": [
                    {
                        "id": "lemonade",
                        "argv": argv if argv is not None else ["python3", "worker.py"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def make_model(
    model_id: str,
    payload: bytes,
    manifest_index: int,
    *,
    size_bytes: int | None = None,
    name: str | None = None,
) -> object:
    display_name = model_id.replace(".", "-")
    return batcher.ModelSpec(
        id=model_id,
        name=name or f"runtime:{display_name}",
        directory=f"dir-{display_name}",
        repository=f"owner/{display_name}",
        revision=f"{manifest_index + 1:040x}",
        filename=f"{display_name}.gguf",
        size_bytes=len(payload) if size_bytes is None else size_bytes,
        sha256=hashlib.sha256(payload).hexdigest(),
        manifest_index=manifest_index,
    )


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.stream = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class ValidationAndPlanningTest(unittest.TestCase):
    def test_manifest_keeps_runtime_name_opaque_and_directory_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            write_manifest(path, [make_manifest_entry()])
            model = batcher.load_manifest(path)[0]

        self.assertEqual(model.name, "runtime name is opaque")
        self.assertEqual(model.directory, "Model-A")
        self.assertNotEqual(model.name, model.directory)

    def test_manifest_rejects_unsafe_and_malformed_fields(self) -> None:
        cases = [
            ({"directory": "../escape"}, "directory"),
            ({"directory": "nested/model"}, "directory"),
            ({"filename": "../model.gguf"}, "filename"),
            ({"repository": "too/many/parts"}, "repository"),
            ({"revision": "main"}, "immutable"),
            ({"size_bytes": True}, "positive integer"),
            ({"sha256": "A" * 64}, "lowercase"),
            ({"name": "bad\nname"}, "control"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            for changes, message in cases:
                with self.subTest(changes=changes):
                    write_manifest(path, [make_manifest_entry(**changes)])
                    with self.assertRaisesRegex(batcher.ManifestError, message):
                        batcher.load_manifest(path)

            entry = make_manifest_entry()
            entry["unexpected"] = 1
            write_manifest(path, [entry])
            with self.assertRaisesRegex(batcher.ManifestError, "unexpected"):
                batcher.load_manifest(path)

    def test_manifest_rejects_duplicate_ids_names_directories_and_json_keys(self) -> None:
        variants = [
            ({"name": "other", "directory": "Other"}, "Duplicate model id"),
            ({"id": "model-b", "directory": "Other"}, "Duplicate runtime"),
            ({"id": "model-b", "name": "other"}, "Duplicate model directory"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            for changes, message in variants:
                with self.subTest(changes=changes):
                    write_manifest(path, [make_manifest_entry(), make_manifest_entry(**changes)])
                    with self.assertRaisesRegex(batcher.ManifestError, message):
                        batcher.load_manifest(path)

            path.write_text(
                '{"schema_version":1,"schema_version":1,"models":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(batcher.ManifestError, "duplicate JSON key"):
                batcher.load_manifest(path)

    def test_selection_rejects_duplicate_and_unknown_ids(self) -> None:
        manifest = [make_model("model-a", b"a", 0)]
        with self.assertRaisesRegex(batcher.ManifestError, "Duplicate requested"):
            batcher.select_models(manifest, ["model-a", "model-a"])
        with self.assertRaisesRegex(batcher.ManifestError, "Unknown requested"):
            batcher.select_models(manifest, ["missing"])

    def test_benchmark_spec_requires_one_clean_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "benchmark.json"
            write_benchmark_spec(path, ["python3", "worker.py", "--fixed", "value"])
            loaded = batcher.load_benchmark_spec(path)
            self.assertEqual(loaded.id, "lemonade")
            self.assertEqual(
                loaded.argv,
                ("python3", "worker.py", "--fixed", "value"),
            )

            for argv, message in [
                ([], "non-empty"),
                (["python3", 3], r"argv\[1\]"),
                (["python3", "worker.py", "--models"], "batch-managed"),
            ]:
                with self.subTest(argv=argv):
                    write_benchmark_spec(path, argv)
                    with self.assertRaisesRegex(batcher.BenchmarkSpecError, message):
                        batcher.load_benchmark_spec(path)

            path.write_text(
                json.dumps({"schema_version": 1, "benchmarks": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(batcher.BenchmarkSpecError, "exactly one"):
                batcher.load_benchmark_spec(path)

    def test_manifest_produces_expected_default_and_reduced_space_batches(self) -> None:
        models = batcher.load_manifest(MANIFEST_PATH)
        default_capacity = batcher.calculate_batch_capacity(
            40.0, free_bytes=100 * batcher.GIB
        )
        self.assertEqual(default_capacity, 38 * batcher.GIB)
        self.assertEqual(
            [
                [model.id for model in group]
                for group in batcher.plan_batches(models, default_capacity)
            ],
            [
                ["qwen3-30b-a3b-instruct-2507", "qwen3.8-27b"],
                ["mistral-small-3.2-24b", "gemma-4-12b", "llama-3.1-8b"],
            ],
        )

        reduced_capacity = batcher.calculate_batch_capacity(
            40.0, free_bytes=32 * batcher.GIB
        )
        self.assertEqual(reduced_capacity, 30 * batcher.GIB)
        self.assertEqual(
            [
                [model.id for model in group]
                for group in batcher.plan_batches(models, reduced_capacity)
            ],
            [
                ["qwen3-30b-a3b-instruct-2507"],
                ["qwen3.8-27b", "mistral-small-3.2-24b"],
                ["gemma-4-12b", "llama-3.1-8b"],
            ],
        )

    def test_disk_plan_rejects_oversize_and_insufficient_space(self) -> None:
        model = make_model("oversize", b"payload", 0, size_bytes=101)
        with self.assertRaisesRegex(batcher.DiskPlanError, "exceed"):
            batcher.plan_batches([model], 100)
        with self.assertRaisesRegex(batcher.DiskPlanError, "reserved"):
            batcher.calculate_batch_capacity(
                40.0, free_bytes=batcher.RESERVED_BYTES
            )
        with self.assertRaisesRegex(batcher.DiskPlanError, "greater than 2"):
            batcher.calculate_batch_capacity(2.0, free_bytes=100 * batcher.GIB)

    def test_remote_validation_rejects_revision_size_and_hash_drift(self) -> None:
        model = make_model("model-a", b"payload", 0)

        def metadata(
            *, revision: str | None = None, size: int | None = None, sha256: str | None = None
        ) -> FakeResponse:
            payload = {
                "sha": revision if revision is not None else model.revision,
                "siblings": [
                    {
                        "rfilename": model.filename,
                        "lfs": {
                            "size": size if size is not None else model.size_bytes,
                            "sha256": sha256 if sha256 is not None else model.sha256,
                        },
                    }
                ],
            }
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        batcher.validate_remote_model(
            model, urlopen=lambda *_args, **_kwargs: metadata()
        )
        for response, message in [
            (metadata(revision="f" * 40), "revision mismatch"),
            (metadata(size=model.size_bytes + 1), "size mismatch"),
            (metadata(sha256="0" * 64), "SHA-256 mismatch"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(batcher.RemoteValidationError, message):
                    batcher.validate_remote_model(
                        model, urlopen=lambda *_args, response=response, **_kwargs: response
                    )

    def test_remote_validation_retries_transient_metadata_failure(self) -> None:
        model = make_model("model-a", b"payload", 0)
        payload = json.dumps(
            {
                "sha": model.revision,
                "siblings": [
                    {
                        "rfilename": model.filename,
                        "lfs": {
                            "size": model.size_bytes,
                            "sha256": model.sha256,
                        },
                    }
                ],
            }
        ).encode("utf-8")
        calls = 0

        def urlopen(*_args: object, **_kwargs: object) -> FakeResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient metadata failure")
            return FakeResponse(payload)

        batcher.validate_remote_model(
            model,
            urlopen=urlopen,
            attempts=2,
            retry_delay=0,
        )
        self.assertEqual(calls, 2)


class DownloadTest(unittest.TestCase):
    def test_fresh_download_has_no_range_and_is_published_atomically(self) -> None:
        data = b"fresh-model-payload"
        model = make_model("fresh", data, 0)
        requests: list[object] = []

        def urlopen(request: object, **_kwargs: object) -> FakeResponse:
            requests.append(request)
            return FakeResponse(data, headers={"Content-Length": str(len(data))})

        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory) / "models"
            destination = models_dir / model.directory / model.filename
            real_replace = os.replace
            replace_calls: list[tuple[Path, Path]] = []

            def checked_replace(source: object, target: object) -> None:
                source_path = Path(source)
                target_path = Path(target)
                replace_calls.append((source_path, target_path))
                self.assertTrue(source_path.is_file())
                self.assertEqual(source_path.read_bytes(), data)
                self.assertFalse(target_path.exists())
                self.assertNotIn(".partial", source_path.name)
                real_replace(source_path, target_path)

            with mock.patch.object(batcher.os, "replace", side_effect=checked_replace):
                actual = batcher.download_model(
                    model,
                    models_dir,
                    urlopen=urlopen,
                    attempts=1,
                    retry_delay=0,
                )

            self.assertEqual(actual, destination)
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(len(replace_calls), 1)
            self.assertIsNone(requests[0].get_header("Range"))

    def test_truncated_response_retry_restarts_complete_transfer(self) -> None:
        data = b"abcdefghij"
        model = make_model("truncated", data, 0)
        requests: list[object] = []

        def urlopen(request: object, **_kwargs: object) -> FakeResponse:
            requests.append(request)
            payload = data[:4] if len(requests) == 1 else data
            return FakeResponse(
                payload,
                headers={"Content-Length": str(len(data))},
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = batcher.download_model(
                model,
                Path(temporary_directory) / "models",
                urlopen=urlopen,
                attempts=2,
                retry_delay=0,
            )
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request.get_header("Range") is None for request in requests))

    def test_partial_http_response_is_never_resumed(self) -> None:
        data = b"complete"
        model = make_model("partial-http", data, 0)
        requests: list[object] = []

        def urlopen(request: object, **_kwargs: object) -> FakeResponse:
            requests.append(request)
            if len(requests) == 1:
                return FakeResponse(data[3:], status=206)
            return FakeResponse(data, headers={"Content-Length": str(len(data))})

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = batcher.download_model(
                model,
                Path(temporary_directory) / "models",
                urlopen=urlopen,
                attempts=2,
                retry_delay=0,
            )
            self.assertEqual(destination.read_bytes(), data)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request.get_header("Range") is None for request in requests))

    def test_size_and_hash_failures_do_not_publish_or_replace_destination(self) -> None:
        expected = b"abcdefghij"
        corrupt = b"abcdEfghij"
        model = make_model("mismatch", expected, 0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory) / "models"
            destination = models_dir / model.directory / model.filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"previous")

            with self.assertRaisesRegex(batcher.DownloadError, "SHA-256 mismatch"):
                batcher.download_model(
                    model,
                    models_dir,
                    urlopen=lambda *_args, **_kwargs: FakeResponse(
                        corrupt, headers={"Content-Length": str(len(corrupt))}
                    ),
                    attempts=1,
                    retry_delay=0,
                )
            self.assertEqual(destination.read_bytes(), b"previous")

            with self.assertRaisesRegex(batcher.DownloadError, "length mismatch"):
                batcher.download_model(
                    model,
                    models_dir,
                    urlopen=lambda *_args, **_kwargs: FakeResponse(
                        expected, headers={"Content-Length": "2"}
                    ),
                    attempts=1,
                    retry_delay=0,
                )
            self.assertEqual(destination.read_bytes(), b"previous")

    def test_download_batch_uses_at_most_two_concurrent_transfers(self) -> None:
        models = [make_model(f"model-{index}", bytes([index]), index) for index in range(4)]
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def download(model: object, _models_dir: Path) -> Path:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return Path(model.filename)

        with mock.patch.object(batcher, "download_model", side_effect=download):
            resident, failures = batcher.download_batch(models, Path("unused"))
        self.assertEqual(resident, models)
        self.assertEqual(failures, [])
        self.assertEqual(maximum_active, 2)


class OrchestrationTest(unittest.TestCase):
    def make_args(self, root: Path, model_ids: list[str]) -> argparse.Namespace:
        benchmark_spec = root / "benchmark.json"
        manifest = root / "manifest.json"
        write_benchmark_spec(
            benchmark_spec,
            ["python3", "worker.py", "--fixed", "value"],
        )
        write_manifest(manifest, [make_manifest_entry()])
        return argparse.Namespace(
            benchmark_spec=benchmark_spec,
            model_manifest=manifest,
            models=model_ids,
            work_root=root / "owned-work",
            max_disk_gib=40.0,
        )

    def test_run_appends_exact_batch_arguments_continues_and_cleans(self) -> None:
        models = [
            make_model("first", b"a", 0, size_bytes=6, name="opaque first"),
            make_model("second", b"b", 1, size_bytes=4, name="extra.Second"),
            make_model("third", b"c", 2, size_bytes=5, name="third-runtime"),
        ]
        commands: list[list[str]] = []

        def download_batch(
            resident: list[object], models_dir: Path
        ) -> tuple[list[object], list[dict[str, object]]]:
            for model in resident:
                model_dir = models_dir / model.directory
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / model.filename).write_bytes(b"resident")
            return list(resident), []

        def run_command(command: list[str], **_kwargs: object) -> SimpleNamespace:
            commands.append(command)
            models_dir = Path(command[command.index("--models-dir") + 1])
            self.assertTrue(models_dir.is_dir())
            return SimpleNamespace(returncode=7 if len(commands) == 1 else 0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root, [model.id for model in models])
            with (
                mock.patch.object(batcher, "load_manifest", return_value=models),
                mock.patch.object(
                    batcher, "validate_remote_models", return_value=(models, [])
                ),
                mock.patch.object(
                    batcher.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * batcher.GIB),
                ),
                mock.patch.object(
                    batcher, "calculate_batch_capacity", return_value=10
                ),
                mock.patch.object(
                    batcher, "download_batch", side_effect=download_batch
                ),
                mock.patch.object(batcher.subprocess, "run", side_effect=run_command),
            ):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    result = batcher.run(args)

            self.assertEqual(result, 1)
            self.assertEqual(len(commands), 2)
            base = ["python3", "worker.py", "--fixed", "value"]
            for batch_number, command in enumerate(commands, start=1):
                self.assertEqual(command[: len(base)], base)
                self.assertEqual(command[len(base) : len(base) + 3], [
                    "--batched",
                    "--batch-number",
                    str(batch_number),
                ])
            self.assertEqual(
                commands[0][commands[0].index("--models") + 1 :],
                [models[0].name],
            )
            self.assertEqual(
                commands[1][commands[1].index("--models") + 1 :],
                [models[2].name, models[1].name],
            )
            self.assertIn("benchmark exit", stderr.getvalue())
            self.assertFalse(args.work_root.exists())

    def test_failed_download_is_skipped_and_other_resident_model_runs(self) -> None:
        models = [
            make_model("first", b"a", 0, size_bytes=6),
            make_model("second", b"b", 1, size_bytes=4),
        ]
        commands: list[list[str]] = []

        def download_batch(
            _batch: list[object], models_dir: Path
        ) -> tuple[list[object], list[dict[str, object]]]:
            model = models[1]
            (models_dir / model.directory).mkdir(parents=True)
            return [model], [
                {"stage": "download", "model": models[0].id, "message": "failed"}
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root, [model.id for model in models])
            with (
                mock.patch.object(batcher, "load_manifest", return_value=models),
                mock.patch.object(
                    batcher, "validate_remote_models", return_value=(models, [])
                ),
                mock.patch.object(
                    batcher.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * batcher.GIB),
                ),
                mock.patch.object(
                    batcher, "calculate_batch_capacity", return_value=10
                ),
                mock.patch.object(
                    batcher, "download_batch", side_effect=download_batch
                ),
                mock.patch.object(
                    batcher.subprocess,
                    "run",
                    side_effect=lambda command, **_kwargs: (
                        commands.append(command) or SimpleNamespace(returncode=0)
                    ),
                ),
            ):
                with redirect_stderr(io.StringIO()):
                    result = batcher.run(args)

        self.assertEqual(result, 1)
        self.assertEqual(len(commands), 1)
        self.assertEqual(
            commands[0][commands[0].index("--models") + 1 :],
            [models[1].name],
        )

    def test_all_failed_batch_is_skipped_but_later_batch_runs(self) -> None:
        models = [
            make_model("first", b"a", 0, size_bytes=6),
            make_model("second", b"b", 1, size_bytes=5),
        ]
        download_calls = 0
        commands: list[list[str]] = []

        def download_batch(
            batch: list[object], models_dir: Path
        ) -> tuple[list[object], list[dict[str, object]]]:
            nonlocal download_calls
            download_calls += 1
            if download_calls == 1:
                return [], [
                    {"stage": "download", "model": batch[0].id, "message": "failed"}
                ]
            (models_dir / batch[0].directory).mkdir(parents=True)
            return batch, []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root, [model.id for model in models])
            with (
                mock.patch.object(batcher, "load_manifest", return_value=models),
                mock.patch.object(
                    batcher, "validate_remote_models", return_value=(models, [])
                ),
                mock.patch.object(
                    batcher.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * batcher.GIB),
                ),
                mock.patch.object(
                    batcher, "calculate_batch_capacity", return_value=6
                ),
                mock.patch.object(
                    batcher, "download_batch", side_effect=download_batch
                ),
                mock.patch.object(
                    batcher.subprocess,
                    "run",
                    side_effect=lambda command, **_kwargs: (
                        commands.append(command) or SimpleNamespace(returncode=0)
                    ),
                ),
            ):
                with redirect_stderr(io.StringIO()):
                    result = batcher.run(args)

        self.assertEqual(result, 1)
        self.assertEqual(download_calls, 2)
        self.assertEqual(len(commands), 1)
        self.assertIn("2", commands[0])

    def test_cleanup_failure_is_the_only_batch_failure_that_aborts_later_batches(self) -> None:
        models = [
            make_model("first", b"a", 0, size_bytes=6),
            make_model("second", b"b", 1, size_bytes=5),
        ]
        download_calls = 0

        def download_batch(
            batch: list[object], models_dir: Path
        ) -> tuple[list[object], list[dict[str, object]]]:
            nonlocal download_calls
            download_calls += 1
            (models_dir / batch[0].directory).mkdir(parents=True)
            return batch, []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = self.make_args(root, [model.id for model in models])
            with (
                mock.patch.object(batcher, "load_manifest", return_value=models),
                mock.patch.object(
                    batcher, "validate_remote_models", return_value=(models, [])
                ),
                mock.patch.object(
                    batcher.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=100 * batcher.GIB),
                ),
                mock.patch.object(
                    batcher, "calculate_batch_capacity", return_value=6
                ),
                mock.patch.object(
                    batcher, "download_batch", side_effect=download_batch
                ),
                mock.patch.object(
                    batcher.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                mock.patch.object(
                    batcher, "cleanup_batch", side_effect=OSError("locked")
                ),
            ):
                with redirect_stderr(io.StringIO()):
                    result = batcher.run(args)

            self.assertEqual(result, 1)
            self.assertEqual(download_calls, 1)
            self.assertTrue(args.work_root.exists())

    def test_cleanup_rejects_changed_root_and_detects_no_op_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "owned"
            root.mkdir()
            batch_dir = root / "batch-0001"
            batch_dir.mkdir()

            with mock.patch.object(batcher.shutil, "rmtree", return_value=None):
                with self.assertRaisesRegex(
                    batcher.BatchBenchmarkError, "did not remove"
                ):
                    batcher.cleanup_batch(batch_dir, root.resolve())

            batcher.shutil.rmtree(batch_dir)
            target = Path(temporary_directory) / "replacement"
            target.mkdir()
            root.rmdir()
            root.symlink_to(target, target_is_directory=True)
            changed_batch = root / "batch-0002"
            with self.assertRaisesRegex(
                batcher.BatchBenchmarkError, "work root changed"
            ):
                batcher.cleanup_batch(changed_batch, root)

    def test_parse_args_exposes_only_generic_interface_and_default(self) -> None:
        parsed = batcher.parse_args(
            [
                "--benchmark-spec",
                "benchmark.json",
                "--model-manifest",
                "manifest.json",
                "--models",
                "one",
                "two",
                "--work-root",
                "work",
            ]
        )
        self.assertEqual(
            set(vars(parsed)),
            {
                "benchmark_spec",
                "model_manifest",
                "models",
                "work_root",
                "max_disk_gib",
            },
        )
        self.assertEqual(parsed.max_disk_gib, 40.0)


if __name__ == "__main__":
    unittest.main()
