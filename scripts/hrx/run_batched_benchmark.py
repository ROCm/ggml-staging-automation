#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run a command over verified, disk-bounded batches of downloaded models."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


GIB = 1024**3
RESERVED_BYTES = 2 * GIB
MAX_CONCURRENT_DOWNLOADS = 2
REMOTE_VALIDATION_ATTEMPTS = 3
DOWNLOAD_ATTEMPTS = 3
HF_BASE_URL = "https://huggingface.co"

ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
LOCAL_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?$"
)
REPOSITORY_PART_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BatchBenchmarkError(RuntimeError):
    """Base error for an invalid or unsafe batched benchmark request."""


class ManifestError(BatchBenchmarkError):
    """Raised when the model manifest is malformed or unsafe."""


class BenchmarkSpecError(BatchBenchmarkError):
    """Raised when the benchmark command specification is malformed."""


class DiskPlanError(BatchBenchmarkError):
    """Raised when the selected models cannot fit the disk budget."""


class RemoteValidationError(BatchBenchmarkError):
    """Raised when a pinned remote file differs from the manifest."""


class DownloadError(BatchBenchmarkError):
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
    manifest_index: int

    @property
    def api_url(self) -> str:
        repository = urllib.parse.quote(self.repository, safe="/")
        return (
            f"{HF_BASE_URL}/api/models/{repository}/revision/"
            f"{self.revision}?blobs=true"
        )

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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, description: str, error_type: type[BatchBenchmarkError]) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise error_type(f"Could not load {description} {path}: {exc}") from exc


def _require_object(
    value: Any,
    context: str,
    error_type: type[BatchBenchmarkError],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{context} must be a JSON object")
    return value


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    context: str,
    error_type: type[BatchBenchmarkError],
) -> None:
    if set(value) == expected:
        return
    details: list[str] = []
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing:
        details.append(f"missing {missing}")
    if unexpected:
        details.append(f"unexpected {unexpected}")
    raise error_type(f"{context} has invalid fields: {', '.join(details)}")


def _require_string(
    value: dict[str, Any],
    field_name: str,
    context: str,
    error_type: type[BatchBenchmarkError],
) -> str:
    result = value.get(field_name)
    if not isinstance(result, str) or not result:
        raise error_type(f"{context}.{field_name} must be a non-empty string")
    if "\0" in result:
        raise error_type(f"{context}.{field_name} must not contain a NUL byte")
    return result


def _has_control_characters(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cs"} for character in value)


def _validate_local_name(value: str, field: str, context: str) -> None:
    if (
        not LOCAL_NAME_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or ".." in value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ManifestError(f"{context}.{field} must be a safe local basename")


def _validate_repository(repository: str, context: str) -> None:
    parts = repository.split("/")
    if len(parts) != 2 or any(
        not REPOSITORY_PART_PATTERN.fullmatch(part) or ".." in part for part in parts
    ):
        raise ManifestError(
            f"{context}.repository must be a safe owner/repository name"
        )


def load_manifest(path: Path) -> list[ModelSpec]:
    """Load and strictly validate an ordered model manifest."""
    data = _load_json(path, "model manifest", ManifestError)
    root = _require_object(data, "manifest", ManifestError)
    _require_exact_fields(
        root, {"schema_version", "models"}, "manifest", ManifestError
    )
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise ManifestError("manifest.schema_version must equal 1")

    entries = root.get("models")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest.models must be a non-empty array")

    expected_fields = {
        "id",
        "name",
        "directory",
        "repository",
        "revision",
        "filename",
        "size_bytes",
        "sha256",
    }
    models: list[ModelSpec] = []
    ids: set[str] = set()
    names: set[str] = set()
    directories: set[str] = set()

    for index, raw_entry in enumerate(entries):
        context = f"manifest.models[{index}]"
        entry = _require_object(raw_entry, context, ManifestError)
        _require_exact_fields(entry, expected_fields, context, ManifestError)

        model_id = _require_string(entry, "id", context, ManifestError)
        name = _require_string(entry, "name", context, ManifestError)
        directory = _require_string(entry, "directory", context, ManifestError)
        repository = _require_string(entry, "repository", context, ManifestError)
        revision = _require_string(entry, "revision", context, ManifestError)
        filename = _require_string(entry, "filename", context, ManifestError)
        sha256 = _require_string(entry, "sha256", context, ManifestError)
        size_bytes = entry.get("size_bytes")

        if not ID_PATTERN.fullmatch(model_id) or ".." in model_id:
            raise ManifestError(f"{context}.id is unsafe: {model_id!r}")
        # `name` is intentionally opaque. It is passed to the configured command
        # exactly as supplied and is never used to form a path.
        if _has_control_characters(name):
            raise ManifestError(f"{context}.name must not contain control characters")
        _validate_local_name(directory, "directory", context)
        _validate_local_name(filename, "filename", context)
        _validate_repository(repository, context)
        if not REVISION_PATTERN.fullmatch(revision):
            raise ManifestError(
                f"{context}.revision must be an immutable 40-character commit"
            )
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ManifestError(f"{context}.size_bytes must be a positive integer")
        if not SHA256_PATTERN.fullmatch(sha256):
            raise ManifestError(
                f"{context}.sha256 must be a lowercase 64-character digest"
            )
        if model_id in ids:
            raise ManifestError(f"Duplicate model id in manifest: {model_id}")
        if name in names:
            raise ManifestError(f"Duplicate runtime model name in manifest: {name}")
        if directory in directories:
            raise ManifestError(f"Duplicate model directory in manifest: {directory}")

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
                manifest_index=index,
            )
        )
    return models


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    """Load the single-command version-one benchmark specification."""
    data = _load_json(path, "benchmark specification", BenchmarkSpecError)
    root = _require_object(data, "benchmark specification", BenchmarkSpecError)
    _require_exact_fields(
        root,
        {"schema_version", "benchmarks"},
        "benchmark specification",
        BenchmarkSpecError,
    )
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        raise BenchmarkSpecError("benchmark specification schema_version must equal 1")
    entries = root.get("benchmarks")
    if not isinstance(entries, list) or len(entries) != 1:
        raise BenchmarkSpecError(
            "benchmark specification must contain exactly one benchmark"
        )

    context = "benchmark specification.benchmarks[0]"
    entry = _require_object(entries[0], context, BenchmarkSpecError)
    _require_exact_fields(entry, {"id", "argv"}, context, BenchmarkSpecError)
    benchmark_id = _require_string(entry, "id", context, BenchmarkSpecError)
    if not ID_PATTERN.fullmatch(benchmark_id) or ".." in benchmark_id:
        raise BenchmarkSpecError(f"{context}.id is unsafe: {benchmark_id!r}")
    argv = entry.get("argv")
    if not isinstance(argv, list) or not argv:
        raise BenchmarkSpecError(f"{context}.argv must be a non-empty string array")
    for index, argument in enumerate(argv):
        if (
            not isinstance(argument, str)
            or not argument
            or _has_control_characters(argument)
        ):
            raise BenchmarkSpecError(
                f"{context}.argv[{index}] must be a non-empty string without "
                "control characters"
            )
    managed_arguments = {"--batched", "--batch-number", "--models-dir", "--models"}
    present_managed = [argument for argument in argv if argument in managed_arguments]
    if present_managed:
        raise BenchmarkSpecError(
            f"{context}.argv must not contain batch-managed arguments: "
            f"{', '.join(sorted(set(present_managed)))}"
        )
    return BenchmarkSpec(id=benchmark_id, argv=tuple(argv))


def select_models(manifest: list[ModelSpec], requested_ids: list[str]) -> list[ModelSpec]:
    """Select requested model IDs while retaining manifest order."""
    if not requested_ids:
        raise ManifestError("At least one model id must be requested")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for model_id in requested_ids:
        if model_id in seen:
            duplicates.add(model_id)
        seen.add(model_id)
    if duplicates:
        raise ManifestError(
            f"Duplicate requested model id(s): {', '.join(sorted(duplicates))}"
        )
    by_id = {model.id: model for model in manifest}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise ManifestError(f"Unknown requested model id(s): {', '.join(unknown)}")
    requested = set(requested_ids)
    return [model for model in manifest if model.id in requested]


def _normalize_remote_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().strip('"').lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    return normalized


def validate_remote_model(
    model: ModelSpec,
    *,
    urlopen: Callable[..., Any] | None = None,
    timeout: float = 30.0,
    attempts: int = REMOTE_VALIDATION_ATTEMPTS,
    retry_delay: float = 1.0,
) -> None:
    """Validate the immutable revision and LFS identity before downloading."""
    opener = urlopen or urllib.request.urlopen
    request = urllib.request.Request(
        model.api_url,
        headers={"Accept": "application/json", "User-Agent": "hrx-benchmark/1"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout=timeout) as response:
                payload = response.read()
        except (OSError, http.client.HTTPException) as exc:
            if attempt == attempts:
                raise RemoteValidationError(
                    f"Could not read pinned metadata for {model.id}: {exc}"
                ) from exc
            log(
                f"Remote validation attempt {attempt}/{attempts} failed for "
                f"{model.id}: {exc}; retrying"
            )
            if retry_delay:
                time.sleep(retry_delay * attempt)
        else:
            break
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteValidationError(
            f"Could not parse pinned metadata for {model.id}: {exc}"
        ) from exc

    if not isinstance(data, dict) or data.get("sha") != model.revision:
        actual_revision = data.get("sha") if isinstance(data, dict) else None
        raise RemoteValidationError(
            f"Pinned revision mismatch for {model.id}: expected {model.revision}, "
            f"got {actual_revision!r}"
        )
    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        raise RemoteValidationError(
            f"Pinned metadata for {model.id} has no file listing"
        )
    matches = [
        sibling
        for sibling in siblings
        if isinstance(sibling, dict) and sibling.get("rfilename") == model.filename
    ]
    if len(matches) != 1:
        raise RemoteValidationError(
            f"Pinned revision for {model.id} does not contain exactly one "
            f"{model.filename!r}"
        )
    sibling = matches[0]
    lfs = sibling.get("lfs")
    if not isinstance(lfs, dict):
        raise RemoteValidationError(
            f"Pinned file for {model.id} has no verifiable LFS metadata"
        )
    remote_size = lfs.get("size")
    remote_sha = _normalize_remote_sha(lfs.get("sha256", lfs.get("oid")))
    if remote_size != model.size_bytes:
        raise RemoteValidationError(
            f"Pinned size mismatch for {model.id}: expected {model.size_bytes}, "
            f"got {remote_size!r}"
        )
    if remote_sha != model.sha256:
        raise RemoteValidationError(
            f"Pinned SHA-256 mismatch for {model.id}: expected {model.sha256}, "
            f"got {remote_sha!r}"
        )


def _failure(stage: str, message: str, **context: Any) -> dict[str, Any]:
    return {"stage": stage, **context, "message": message}


def validate_remote_models(
    models: list[ModelSpec],
) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    """Validate all selected identities before allowing the first download."""
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_DOWNLOADS, len(models))
    ) as executor:
        futures = [executor.submit(validate_remote_model, model) for model in models]
        valid: list[ModelSpec] = []
        failures: list[dict[str, Any]] = []
        for model, future in zip(models, futures):
            try:
                future.result()
            except RemoteValidationError as exc:
                log(f"Remote validation failed for {model.id}: {exc}")
                failures.append(
                    _failure("remote validation", str(exc), model=model.id)
                )
            else:
                valid.append(model)
                log(f"Verified pinned remote identity for {model.id}")
    return valid, failures


def calculate_batch_capacity(max_disk_gib: float, free_bytes: int) -> int:
    """Reserve 2 GiB and cap model residency by policy and current free space."""
    if not math.isfinite(max_disk_gib) or max_disk_gib <= 2:
        raise DiskPlanError("--max-disk-gib must be finite and greater than 2")
    maximum_bytes = int(max_disk_gib * GIB)
    if free_bytes <= RESERVED_BYTES:
        raise DiskPlanError(
            f"Only {free_bytes} bytes are free; {RESERVED_BYTES} bytes are reserved"
        )
    capacity = min(maximum_bytes - RESERVED_BYTES, free_bytes - RESERVED_BYTES)
    return capacity


def plan_batches(models: list[ModelSpec], capacity_bytes: int) -> list[list[ModelSpec]]:
    """Plan deterministic next-fit batches in descending model-size order."""
    oversized = [model for model in models if model.size_bytes > capacity_bytes]
    if oversized:
        details = ", ".join(
            f"{model.id} ({model.size_bytes} bytes)" for model in oversized
        )
        raise DiskPlanError(
            f"Selected model(s) exceed the {capacity_bytes}-byte batch capacity: "
            f"{details}"
        )

    ordered = sorted(models, key=lambda model: (-model.size_bytes, model.manifest_index))
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(
    model: ModelSpec,
    temporary_path: Path,
    *,
    urlopen: Callable[..., Any],
    timeout: float,
) -> None:
    """Transfer one complete response from byte zero into a fresh file."""
    request = urllib.request.Request(
        model.download_url,
        headers={"User-Agent": "hrx-benchmark/1"},
    )
    with urlopen(request, timeout=timeout) as response:
        status = _response_status(response)
        if status != 200:
            raise DownloadError(
                f"Download for {model.id} returned HTTP {status}; expected a full response"
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError) as exc:
                raise DownloadError(
                    f"Download for {model.id} has invalid Content-Length "
                    f"{content_length!r}"
                ) from exc
            if declared_size != model.size_bytes:
                raise DownloadError(
                    f"Download length mismatch for {model.id}: expected "
                    f"{model.size_bytes}, got {declared_size}"
                )

        with temporary_path.open("wb") as output_file:
            while chunk := response.read(1024 * 1024):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())

    actual_size = temporary_path.stat().st_size
    if actual_size != model.size_bytes:
        raise DownloadError(
            f"Download size mismatch for {model.id}: expected {model.size_bytes} bytes, "
            f"got {actual_size}"
        )
    actual_sha = sha256_file(temporary_path)
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
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / model.filename
    last_error: Exception

    for attempt in range(1, attempts + 1):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=model_dir,
                prefix=f".{model.filename}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
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
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # Exact batch cleanup remains responsible for any file that
                    # cannot be removed after an individual failed attempt.
                    pass
            if attempt < attempts:
                log(
                    f"Download attempt {attempt}/{attempts} failed for {model.id}: "
                    f"{exc}; restarting from byte zero"
                )
                if retry_delay:
                    time.sleep(retry_delay * attempt)

    raise DownloadError(
        f"Could not download {model.id} after {attempts} attempt(s): {last_error}"
    )


def download_batch(
    models: list[ModelSpec], models_dir: Path
) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    """Download at most two models concurrently, preserving planned order."""
    with ThreadPoolExecutor(
        max_workers=min(MAX_CONCURRENT_DOWNLOADS, len(models))
    ) as executor:
        futures = [executor.submit(download_model, model, models_dir) for model in models]
        resident: list[ModelSpec] = []
        failures: list[dict[str, Any]] = []
        for model, future in zip(models, futures):
            try:
                future.result()
            except DownloadError as exc:
                log(f"Download failed for {model.id}: {exc}")
                failures.append(_failure("download", str(exc), model=model.id))
            else:
                resident.append(model)
    return resident, failures


def prepare_work_root(path: Path) -> Path:
    """Claim an absent or empty, non-symlink directory for exact cleanup."""
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BatchBenchmarkError(f"Work root must not be a symlink: {path}")
    resolved = expanded.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden or resolved.parent == resolved:
        raise BatchBenchmarkError(f"Refusing unsafe work root: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise BatchBenchmarkError(f"Work root is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise BatchBenchmarkError(f"Work root must be empty: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def cleanup_batch(batch_dir: Path, work_root: Path) -> None:
    """Remove exactly one owned, directly nested batch directory."""
    if (
        work_root.is_symlink()
        or not work_root.is_dir()
        or work_root.resolve() != work_root
    ):
        raise BatchBenchmarkError(
            f"Owned work root changed before batch cleanup: {work_root}"
        )
    if (
        batch_dir.is_symlink()
        or not batch_dir.is_dir()
        or batch_dir.parent != work_root
        or batch_dir.resolve() != batch_dir
        or re.fullmatch(r"batch-\d{4}", batch_dir.name) is None
    ):
        raise BatchBenchmarkError(f"Refusing unsafe batch cleanup target: {batch_dir}")
    shutil.rmtree(batch_dir)
    if batch_dir.exists() or batch_dir.is_symlink():
        raise BatchBenchmarkError(f"Batch cleanup did not remove target: {batch_dir}")


def cleanup_work_root(work_root: Path) -> None:
    """Remove the verified-empty owned root and confirm its disappearance."""
    if (
        work_root.is_symlink()
        or not work_root.is_dir()
        or work_root.resolve() != work_root
    ):
        raise BatchBenchmarkError(
            f"Owned work root changed before final cleanup: {work_root}"
        )
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


def _summarize_failures(failures: list[dict[str, Any]]) -> None:
    if not failures:
        log("Batched benchmark completed successfully")
        return
    print(f"Failure summary ({len(failures)}):", file=sys.stderr)
    for failure in failures:
        context: list[str] = []
        for key in ("batch", "benchmark", "model"):
            if key in failure:
                context.append(f"{key}={failure[key]}")
        label = failure["stage"]
        if context:
            label += f" ({', '.join(context)})"
        message = " ".join(str(failure["message"]).split())
        if len(message) > 300:
            message = message[:297] + "..."
        print(f"- {label}: {message}", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    """Execute every viable batch and return nonzero after all failures are known."""
    failures: list[dict[str, Any]] = []
    work_root: Path | None = None
    cleanup_exact = True
    completed_batches = 0

    try:
        benchmark = load_benchmark_spec(args.benchmark_spec)
        manifest = load_manifest(args.model_manifest)
        requested = select_models(manifest, args.models)
        work_root = prepare_work_root(args.work_root)

        valid_models, remote_failures = validate_remote_models(requested)
        failures.extend(remote_failures)

        free_bytes = shutil.disk_usage(work_root).free
        capacity_bytes = calculate_batch_capacity(args.max_disk_gib, free_bytes)
        batches = plan_batches(valid_models, capacity_bytes)
        log(
            "Disk capacity: "
            f"limit={args.max_disk_gib:g} GiB, free={_format_bytes(free_bytes)}, "
            f"reserve={_format_bytes(RESERVED_BYTES)}, "
            f"resident capacity={_format_bytes(capacity_bytes)}"
        )
        log(f"Planned {len(batches)} resident batch(es)")
        for batch_number, batch in enumerate(batches, start=1):
            batch_size = sum(model.size_bytes for model in batch)
            log(
                f"Batch {batch_number}: {', '.join(model.id for model in batch)} "
                f"[{_format_bytes(batch_size)}]"
            )

        for batch_number, batch in enumerate(batches, start=1):
            batch_dir = work_root / f"batch-{batch_number:04d}"
            models_dir = batch_dir / "models"
            batch_ids = [model.id for model in batch]
            try:
                batch_dir.mkdir()
                resident, download_failures = download_batch(batch, models_dir)
                failures.extend(
                    {"batch": batch_number, **failure}
                    for failure in download_failures
                )
                if not resident:
                    log(
                        f"Skipping benchmark for batch {batch_number}: no models "
                        "downloaded successfully"
                    )
                else:
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
                            f"Benchmark {benchmark.id} batch {batch_number} "
                            f"failed to run: {exc}"
                        )
                        failures.append(
                            _failure(
                                "benchmark invocation",
                                str(exc),
                                batch=batch_number,
                                benchmark=benchmark.id,
                            )
                        )
                    else:
                        completed_batches += 1
                        log(
                            f"Benchmark {benchmark.id} batch {batch_number} exited "
                            f"with status {completed.returncode}"
                        )
                        if completed.returncode != 0:
                            failures.append(
                                _failure(
                                    "benchmark exit",
                                    f"command exited with status {completed.returncode}",
                                    batch=batch_number,
                                    benchmark=benchmark.id,
                                )
                            )
            except OSError as exc:
                log(f"Batch {batch_number} failed: {exc}")
                failures.append(
                    _failure(
                        "batch",
                        str(exc),
                        batch=batch_number,
                        models=", ".join(batch_ids),
                    )
                )
            finally:
                try:
                    cleanup_batch(batch_dir, work_root)
                except (BatchBenchmarkError, OSError) as exc:
                    cleanup_exact = False
                    failures.append(
                        _failure("cleanup", str(exc), batch=batch_number)
                    )
                    log(
                        f"Batch {batch_number} cleanup failed; aborting later batches "
                        "because the disk bound can no longer be guaranteed"
                    )
                else:
                    log(f"Cleaned batch {batch_number}: {batch_dir}")
            if not cleanup_exact:
                break
    except (BatchBenchmarkError, OSError) as exc:
        failures.append(_failure("setup", str(exc)))
    finally:
        if work_root is not None and cleanup_exact:
            try:
                cleanup_work_root(work_root)
            except (BatchBenchmarkError, OSError) as exc:
                failures.append(
                    _failure("cleanup", f"Could not remove work root {work_root}: {exc}")
                )
            else:
                log(f"Cleaned work root {work_root}")

    if completed_batches:
        log(f"Completed {completed_batches} benchmark batch invocation(s)")
    _summarize_failures(failures)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
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
