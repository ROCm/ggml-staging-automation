#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run commands over verified, disk-bounded batches of downloaded models."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


GIB = 1024**3
RESERVED_BYTES = 2 * GIB
MAX_CONCURRENT_DOWNLOADS = 2
DOWNLOAD_ATTEMPTS = 3
HF_BASE_URL = "https://huggingface.co"


class BatchBenchmarkError(RuntimeError):
    """Base error for an invalid or unsafe batched benchmark request."""


class DownloadError(RuntimeError):
    """Raised when a model cannot be downloaded and verified."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    directory: str
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str

    @property
    def download_url(self) -> str:
        repository = urllib.parse.quote(self.repository, safe="/")
        filename = urllib.parse.quote(self.filename, safe="")
        return f"{HF_BASE_URL}/{repository}/resolve/{self.revision}/{filename}"


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    argv: tuple[str, ...]


def log(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchBenchmarkError(
            f"Could not load {description} {path}: {exc}"
        ) from exc


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchBenchmarkError(f"{context} must be a JSON object")
    return value


def _require_string(
    value: dict[str, Any],
    field_name: str,
    context: str,
) -> str:
    result = value.get(field_name)
    if not isinstance(result, str) or not result:
        raise BatchBenchmarkError(
            f"{context}.{field_name} must be a non-empty string"
        )
    if "\0" in result:
        raise BatchBenchmarkError(
            f"{context}.{field_name} must not contain a NUL byte"
        )
    return result


def _validate_local_name(value: str, field: str, context: str) -> None:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise BatchBenchmarkError(
            f"{context}.{field} must be a safe single path component"
        )


def _validate_repository(repository: str, context: str) -> None:
    parts = repository.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(part in {".", ".."} for part in parts)
    ):
        raise BatchBenchmarkError(
            f"{context}.repository must be an owner/repository pair"
        )


def load_manifest(path: Path) -> list[ModelSpec]:
    """Load and validate an ordered model manifest."""
    root = _require_object(_load_json(path, "model manifest"), "manifest")
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise BatchBenchmarkError("manifest.schema_version must equal 1")

    entries = root.get("models")
    if not isinstance(entries, list) or not entries:
        raise BatchBenchmarkError("manifest.models must be a non-empty array")

    models: list[ModelSpec] = []
    ids: set[str] = set()
    names: set[str] = set()
    directories: set[str] = set()
    for index, raw_entry in enumerate(entries):
        context = f"manifest.models[{index}]"
        entry = _require_object(raw_entry, context)

        model_id = _require_string(entry, "id", context)
        name = _require_string(entry, "name", context)
        directory = _require_string(entry, "directory", context)
        repository = _require_string(entry, "repository", context)
        revision = _require_string(entry, "revision", context)
        filename = _require_string(entry, "filename", context)
        sha256 = _require_string(entry, "sha256", context)
        size_bytes = entry.get("size_bytes")

        _validate_local_name(directory, "directory", context)
        _validate_local_name(filename, "filename", context)
        _validate_repository(repository, context)
        if type(size_bytes) is not int or size_bytes <= 0:
            raise BatchBenchmarkError(
                f"{context}.size_bytes must be a positive integer"
            )
        if model_id in ids:
            raise BatchBenchmarkError(
                f"Duplicate model id in manifest: {model_id}"
            )
        if name in names:
            raise BatchBenchmarkError(
                f"Duplicate runtime model name in manifest: {name}"
            )
        if directory in directories:
            raise BatchBenchmarkError(
                f"Duplicate model directory in manifest: {directory}"
            )

        ids.add(model_id)
        names.add(name)
        directories.add(directory)
        models.append(
            ModelSpec(
                id=model_id,
                name=name,
                directory=directory,
                repository=repository,
                revision=revision,
                filename=filename,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
    return models


def load_benchmark_spec(path: Path) -> list[BenchmarkSpec]:
    """Load an ordered version-one benchmark specification."""
    root = _require_object(
        _load_json(path, "benchmark specification"),
        "benchmark specification",
    )
    schema_version = root.get("schema_version")
    version_is_integer = type(schema_version) is int
    version_is_one = schema_version == 1
    version_is_valid = version_is_integer and version_is_one
    if not version_is_valid:
        raise BatchBenchmarkError(
            "benchmark specification schema_version must equal 1"
        )

    entries = root.get("benchmarks")
    entries_are_an_array = isinstance(entries, list)
    entries_are_present = bool(entries)
    entries_are_valid = entries_are_an_array and entries_are_present
    if not entries_are_valid:
        raise BatchBenchmarkError(
            "benchmark specification must contain at least one benchmark"
        )

    benchmark_ids: set[str] = set()
    benchmarks: list[BenchmarkSpec] = []
    managed_arguments = {
        "--batched",
        "--batch-number",
        "--models-dir",
        "--models",
    }
    for entry_index, raw_entry in enumerate(entries):
        context = f"benchmark specification.benchmarks[{entry_index}]"
        entry = _require_object(raw_entry, context)
        benchmark_id = _require_string(entry, "id", context)
        if benchmark_id in benchmark_ids:
            raise BatchBenchmarkError(
                f"Duplicate benchmark id in specification: {benchmark_id}"
            )
        benchmark_ids.add(benchmark_id)

        raw_argv = entry.get("argv")
        argv_is_an_array = isinstance(raw_argv, list)
        argv_is_present = bool(raw_argv)
        argv_is_valid = argv_is_an_array and argv_is_present
        if not argv_is_valid:
            raise BatchBenchmarkError(
                f"{context}.argv must be a non-empty string array"
            )
        argv: list[str] = []
        for argument_index, raw_argument in enumerate(raw_argv):
            argument_context = f"{context}.argv[{argument_index}]"
            argument_is_a_string = isinstance(raw_argument, str)
            argument_is_present = bool(raw_argument)
            argument_is_valid = argument_is_a_string and argument_is_present
            if not argument_is_valid:
                raise BatchBenchmarkError(
                    f"{argument_context} must be a non-empty string"
                )
            argument_has_nul = "\0" in raw_argument
            if argument_has_nul:
                raise BatchBenchmarkError(
                    f"{argument_context} must not contain a NUL byte"
                )
            argv.append(raw_argument)

        present_managed: set[str] = set()
        for argument in argv:
            for managed in managed_arguments:
                is_bare_managed_argument = argument == managed
                is_managed_assignment = argument.startswith(managed + "=")
                is_managed_argument = (
                    is_bare_managed_argument or is_managed_assignment
                )
                if is_managed_argument:
                    present_managed.add(managed)
        if present_managed:
            raise BatchBenchmarkError(
                f"{context}.argv must not contain batch-managed arguments: "
                f"{', '.join(sorted(present_managed))}"
            )
        benchmarks.append(BenchmarkSpec(id=benchmark_id, argv=tuple(argv)))

    return benchmarks


def select_models(
    manifest: list[ModelSpec],
    requested_ids: list[str],
) -> list[ModelSpec]:
    """Select requested model IDs while retaining manifest order."""
    if not requested_ids:
        raise BatchBenchmarkError("At least one model id must be requested")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for model_id in requested_ids:
        if model_id in seen:
            duplicates.add(model_id)
        seen.add(model_id)
    if duplicates:
        raise BatchBenchmarkError(
            f"Duplicate requested model id(s): {', '.join(sorted(duplicates))}"
        )

    by_id = {model.id: model for model in manifest}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise BatchBenchmarkError(
            f"Unknown requested model id(s): {', '.join(unknown)}"
        )
    requested = set(requested_ids)
    return [model for model in manifest if model.id in requested]


def calculate_batch_capacity(max_disk_gib: float, free_bytes: int) -> int:
    """Reserve 2 GiB and cap model residency by policy and current free space."""
    if not math.isfinite(max_disk_gib) or max_disk_gib <= 2:
        raise BatchBenchmarkError(
            "--max-disk-gib must be finite and greater than 2"
        )

    maximum_bytes = int(max_disk_gib * GIB)
    if free_bytes <= RESERVED_BYTES:
        raise BatchBenchmarkError(
            f"Only {free_bytes} bytes are free; "
            f"{RESERVED_BYTES} bytes are reserved"
        )
    return min(
        maximum_bytes - RESERVED_BYTES,
        free_bytes - RESERVED_BYTES,
    )


def plan_batches(
    models: list[ModelSpec],
    capacity_bytes: int,
) -> list[list[ModelSpec]]:
    """Plan deterministic next-fit batches in descending model-size order."""
    oversized = [model for model in models if model.size_bytes > capacity_bytes]
    if oversized:
        details = ", ".join(
            f"{model.id} ({model.size_bytes} bytes)" for model in oversized
        )
        raise BatchBenchmarkError(
            f"Selected model(s) exceed the {capacity_bytes}-byte batch capacity: "
            f"{details}"
        )

    ordered = sorted(
        models,
        key=lambda model: model.size_bytes,
        reverse=True,
    )
    batches: list[list[ModelSpec]] = []
    current: list[ModelSpec] = []
    current_size = 0
    for model in ordered:
        if current and current_size + model.size_bytes > capacity_bytes:
            batches.append(current)
            current = []
            current_size = 0
        current.append(model)
        current_size += model.size_bytes
    if current:
        batches.append(current)
    return batches


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _download_once(
    model: ModelSpec,
    temporary_path: Path,
    *,
    urlopen: Callable[..., Any],
    timeout: float,
) -> None:
    """Stream and verify one complete response from byte zero."""
    request = urllib.request.Request(
        model.download_url,
        headers={"User-Agent": "hrx-benchmark/1"},
    )
    actual_size = 0
    digest = hashlib.sha256()

    with temporary_path.open("wb") as output_file:
        with urlopen(request, timeout=timeout) as response:
            status = _response_status(response)
            if status != 200:
                raise DownloadError(
                    f"Download for {model.id} returned HTTP {status}; "
                    "expected a full response"
                )

            while chunk := response.read(1024 * 1024):
                next_size = actual_size + len(chunk)
                if next_size > model.size_bytes:
                    raise DownloadError(
                        f"Download size exceeded for {model.id}: expected "
                        f"{model.size_bytes} bytes, received more"
                    )
                output_file.write(chunk)
                digest.update(chunk)
                actual_size = next_size

    if actual_size != model.size_bytes:
        raise DownloadError(
            f"Download size mismatch for {model.id}: expected "
            f"{model.size_bytes} bytes, got {actual_size}"
        )

    actual_sha = digest.hexdigest()
    if actual_sha != model.sha256:
        raise DownloadError(
            f"SHA-256 mismatch for {model.id}: expected {model.sha256}, "
            f"got {actual_sha}"
        )


def download_model(
    model: ModelSpec,
    models_dir: Path,
    *,
    urlopen: Callable[..., Any] | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    timeout: float = 120.0,
    retry_delay: float = 1.0,
) -> Path:
    """Download, verify, and atomically publish one model file."""
    opener = urlopen or urllib.request.urlopen
    model_dir = models_dir / model.directory
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(
            f"Could not prepare model directory for {model.id}: {exc}"
        ) from exc
    destination = model_dir / model.filename
    temporary_path = model_dir / f".{model.filename}.tmp"
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            log(f"Downloading {model.id} (attempt {attempt}/{attempts})")
            _download_once(
                model,
                temporary_path,
                urlopen=opener,
                timeout=timeout,
            )
            os.replace(temporary_path, destination)
            log(f"Downloaded and verified {model.id}")
            return destination
        except (DownloadError, OSError, http.client.HTTPException) as exc:
            last_error = exc
            if attempt < attempts:
                log(
                    f"Download attempt {attempt}/{attempts} failed for "
                    f"{model.id}: {exc}; restarting from byte zero"
                )
                if retry_delay:
                    time.sleep(retry_delay * attempt)

    if last_error is None:
        raise DownloadError(
            f"Could not download {model.id}: no attempts were requested"
        )
    raise DownloadError(
        f"Could not download {model.id} after {attempts} attempt(s): "
        f"{last_error}"
    ) from last_error


def download_batch(
    models: list[ModelSpec],
    models_dir: Path,
) -> tuple[list[ModelSpec], list[str]]:
    """Download at most two models concurrently, preserving planned order."""
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_DOWNLOADS, len(models))
    ) as executor:
        futures = [
            executor.submit(download_model, model, models_dir)
            for model in models
        ]
        resident: list[ModelSpec] = []
        failures: list[str] = []
        for model, future in zip(models, futures):
            try:
                future.result()
            except DownloadError as exc:
                message = f"download (model={model.id}): {exc}"
                log(f"Download failed for {model.id}: {exc}")
                failures.append(message)
            else:
                resident.append(model)
    return resident, failures


def prepare_work_root(path: Path) -> Path:
    """Create the workflow-owned work directory."""
    work_root = path.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    return work_root


def cleanup_batch(batch_dir: Path) -> None:
    """Remove a completed batch and confirm its disappearance."""
    shutil.rmtree(batch_dir)
    if batch_dir.exists() or batch_dir.is_symlink():
        raise BatchBenchmarkError(
            f"Batch cleanup did not remove target: {batch_dir}"
        )


def cleanup_work_root(work_root: Path) -> None:
    """Remove the empty work root and confirm its disappearance."""
    work_root.rmdir()
    if work_root.exists() or work_root.is_symlink():
        raise BatchBenchmarkError(
            f"Work-root cleanup did not remove target: {work_root}"
        )


def build_benchmark_command(
    benchmark: BenchmarkSpec,
    *,
    batch_number: int,
    models_dir: Path,
    resident: list[ModelSpec],
) -> list[str]:
    return [
        *benchmark.argv,
        "--batched",
        "--batch-number",
        str(batch_number),
        "--models-dir",
        os.fspath(models_dir),
        "--models",
        *(model.name for model in resident),
    ]


def _format_bytes(value: int) -> str:
    return f"{value} bytes ({value / GIB:.2f} GiB)"


def _summarize_failures(failures: list[str]) -> None:
    if not failures:
        log("Batched benchmark completed successfully")
        return

    print(f"Failure summary ({len(failures)}):", file=sys.stderr)
    for failure in failures:
        print(f"- {' '.join(failure.split())}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    """Execute every viable batch and return nonzero after all failures are known."""
    failures: list[str] = []
    work_root: Path | None = None
    cleanup_exact = True

    try:
        benchmarks = load_benchmark_spec(args.benchmark_spec)
        manifest = load_manifest(args.model_manifest)
        requested = select_models(manifest, args.models)
        work_root = prepare_work_root(args.work_root)

        free_bytes = shutil.disk_usage(work_root).free
        capacity_bytes = calculate_batch_capacity(
            args.max_disk_gib,
            free_bytes,
        )
        batches = plan_batches(requested, capacity_bytes)
        log(
            "Disk capacity: "
            f"limit={args.max_disk_gib:g} GiB, "
            f"free={_format_bytes(free_bytes)}, "
            f"reserve={_format_bytes(RESERVED_BYTES)}, "
            f"resident capacity={_format_bytes(capacity_bytes)}"
        )
        log(f"Planned {len(batches)} resident batch(es)")
        for batch_number, batch in enumerate(batches, start=1):
            batch_size = sum(model.size_bytes for model in batch)
            log(
                f"Batch {batch_number}: "
                f"{', '.join(model.id for model in batch)} "
                f"[{_format_bytes(batch_size)}]"
            )

        for batch_number, batch in enumerate(batches, start=1):
            batch_dir = work_root / f"batch-{batch_number:04d}"
            models_dir = batch_dir / "models"
            try:
                batch_dir.mkdir()
                resident, download_failures = download_batch(
                    batch,
                    models_dir,
                )
                failures.extend(
                    f"batch {batch_number}: {failure}"
                    for failure in download_failures
                )
                if not resident:
                    log(
                        f"Skipping benchmarks for batch {batch_number}: "
                        "no models downloaded successfully"
                    )
                else:
                    for benchmark in benchmarks:
                        command = build_benchmark_command(
                            benchmark,
                            batch_number=batch_number,
                            models_dir=models_dir,
                            resident=resident,
                        )
                        log("++ " + shlex.join(command))
                        try:
                            completed = subprocess.run(command, check=False)
                        except OSError as exc:
                            log(
                                f"Benchmark {benchmark.id} batch "
                                f"{batch_number} failed to run: {exc}"
                            )
                            failures.append(
                                "benchmark invocation "
                                f"(batch={batch_number}, "
                                f"benchmark={benchmark.id}): {exc}"
                            )
                            continue
                        log(
                            f"Benchmark {benchmark.id} batch {batch_number} "
                            f"exited with status {completed.returncode}"
                        )
                        if completed.returncode != 0:
                            failures.append(
                                "benchmark exit "
                                f"(batch={batch_number}, benchmark={benchmark.id}): "
                                "command exited with status "
                                f"{completed.returncode}"
                            )
            except OSError as exc:
                log(f"Batch {batch_number} failed: {exc}")
                failures.append(
                    f"batch (batch={batch_number}, "
                    f"models={', '.join(model.id for model in batch)}): {exc}"
                )
            finally:
                try:
                    cleanup_batch(batch_dir)
                except (BatchBenchmarkError, OSError) as exc:
                    cleanup_exact = False
                    failures.append(
                        f"cleanup (batch={batch_number}): {exc}"
                    )
                    log(
                        f"Batch {batch_number} cleanup failed; "
                        "aborting later batches because the disk bound "
                        "can no longer be guaranteed"
                    )
                else:
                    log(f"Cleaned batch {batch_number}: {batch_dir}")
            if not cleanup_exact:
                break
    except (BatchBenchmarkError, OSError) as exc:
        failures.append(f"setup: {exc}")
    finally:
        if work_root is not None and cleanup_exact:
            try:
                cleanup_work_root(work_root)
            except (BatchBenchmarkError, OSError) as exc:
                failures.append(
                    f"cleanup: Could not remove work root {work_root}: {exc}"
                )
            else:
                log(f"Cleaned work root {work_root}")

    _summarize_failures(failures)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-spec",
        type=Path,
        required=True,
        help="JSON command list; relative argv paths use the current directory",
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, metavar="ID")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--max-disk-gib", type=float, default=40.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (BatchBenchmarkError, OSError) as exc:
        print(f"Batched benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
