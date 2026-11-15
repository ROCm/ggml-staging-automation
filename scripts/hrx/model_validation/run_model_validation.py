#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run model validation checks over a tiered model set within a fixed disk bound.

Terms:

- Model tier: an ordered label in the model manifest ("tiers"). Requesting a
  tier selects that tier plus every earlier one, so tiers are cumulative and
  the manifest order defines the size of the run. The reserved select-only
  tier is excluded; its models require explicit --model-ids.
- Batch: a subset of the selected models that fits on disk at once. Models in a
  batch are downloaded together, checked together, and deleted together.
- Validation check: a command from the model validation specification (for
  example run_perplexity.py). It is invoked once per batch and is expected to
  merge its per-batch results itself.
- Expected result: ``expected_results`` maps check, backend and mode to
  "pass", "fail", or "report-only". Omitted cases expect a pass. Workers read
  these expectations from the manifest; failures and XPASS are strict.
- Test modes: optional ``test-modes`` selects cases independently per check.
  Perplexity remains one comparison per model using the "default" mode.

Why this exists: CI runners for the HRX release model validation checks have
far less free disk than the full model set requires. Instead of pinning model validation to
whatever fits, this driver keeps the model set as data
(``model_validation/hrx/model_manifest.json`` plus
``model_validation/hrx/model_validation_spec.json``), splits it into batches
that respect --max-disk-gib, and guarantees no batch leaks onto the next.

Boundary contract:

- Inputs: --validation-spec (ordered "checks" with "id" and "argv"),
  --model-manifest (ordered "tiers" and "models"), --model-tier, --model-ids, --work-root,
  and --max-disk-gib. Optional --include-checks is a JSON array of IDs
  to run in specification order.
  Both JSON files are fully validated up front; a malformed
  file fails the run before any download.
- Worker invocation: each worker's argv is run verbatim with the batch-managed
  arguments appended: --batched --batch-number N --models-dir DIR
  --models NAME...
  A spec that already carries any managed argument is rejected so a worker cannot be pointed at
  the wrong models or supplied conflicting failure policy.
- Exit status: 0 only if every download, every worker, and every cleanup
  succeeded. Failures are collected and summarized on stderr rather than
  aborting at the first one, so a single flaky download does not hide results
  for the other models.
- Disk invariant: at most --max-disk-gib minus a 2 GiB reserve of model data is
  resident at any time, and never more than is actually free.

How it works:

- Batches are planned deterministically: models are sorted by size descending
  and packed next-fit into the resident capacity. Any single model larger than
  the capacity fails planning immediately.
- Each batch lives in work_root/batch-NNNN/models. Downloads come from
  huggingface.co, at most two at a time, each streamed to a temporary file,
  size-bounded, SHA-256 verified, and then atomically renamed into place. A
  failed attempt restarts from byte zero (no Range resumption) up to three
  times. HTTP errors honor Retry-After seconds or dates; a 429 without a valid
  header backs off for 60 then 120 seconds. A model that never downloads is
  reported and the batch continues with whatever did.
- Workers run only when at least one model in the batch is resident and are
  passed only the resident models, so a partial batch still yields data.
- Batch cleanup runs in a finally block. If a batch directory cannot be removed
  the driver stops scheduling later batches, because the disk bound can no
  longer be guaranteed; the work root is likewise only removed when every batch
  cleanup was exact.
"""

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
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable


GIB = 1024**3
RESERVED_BYTES = 2 * GIB
MAX_CONCURRENT_DOWNLOADS = 2
DOWNLOAD_ATTEMPTS = 3
RATE_LIMIT_RETRY_DELAY = 60.0
HF_BASE_URL = "https://huggingface.co"


class ModelValidationError(RuntimeError):
    """Base error for an invalid or unsafe model validation request."""


class DownloadError(RuntimeError):
    """Raised when a model cannot be downloaded and verified."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: str
    expected_results: dict[str, dict[str, dict[str, str]]]
    test_modes: dict[str, tuple[str, ...]]
    name: str
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str

    def modes(self, check: str) -> tuple[str, ...]:
        """Use the ordinary case when this check has no explicit modes."""
        return self.test_modes.get(check, ("default",))

    def expectation(self, check: str, backend: str, mode: str = "default") -> str:
        return self.expected_results.get(check, {}).get(backend, {}).get(mode, "pass")

    @property
    def download_url(self) -> str:
        repository = urllib.parse.quote(self.repository, safe="/")
        filename = urllib.parse.quote(self.filename, safe="")
        return f"{HF_BASE_URL}/{repository}/resolve/{self.revision}/{filename}"


@dataclass(frozen=True)
class ModelManifest:
    tiers: tuple[str, ...]
    models: tuple[ModelSpec, ...]


@dataclass(frozen=True)
class ValidationCheck:
    id: str
    argv: tuple[str, ...]


def log(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelValidationError(
            f"Could not load {description} {path}: {exc}"
        ) from exc


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelValidationError(f"{context} must be a JSON object")
    return value


def _require_string(
    value: dict[str, Any],
    field_name: str,
    context: str,
) -> str:
    return _require_string_value(
        value.get(field_name),
        f"{context}.{field_name}",
    )


def _require_string_value(value: Any, context: str) -> str:
    value_is_a_string = isinstance(value, str)
    value_is_present = bool(value)
    value_is_valid = value_is_a_string and value_is_present
    if not value_is_valid:
        raise ModelValidationError(f"{context} must be a non-empty string")
    if "\0" in value:
        raise ModelValidationError(f"{context} must not contain a NUL byte")
    return value


def _validate_local_name(value: str, field: str, context: str) -> None:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ModelValidationError(
            f"{context}.{field} must be a safe single path component"
        )


def _validate_repository(repository: str, context: str) -> None:
    parts = repository.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(part in {".", ".."} for part in parts)
    ):
        raise ModelValidationError(
            f"{context}.repository must be an owner/repository pair"
        )


def load_manifest(
    path: Path, *, known_checks: set[str] | None = None
) -> ModelManifest:
    """Load and validate an ordered, cumulative model-tier manifest."""
    root = _require_object(_load_json(path, "model manifest"), "manifest")
    raw_tiers = root.get("tiers")
    tiers_are_an_array = isinstance(raw_tiers, list)
    tiers_are_present = bool(raw_tiers)
    tiers_are_valid = tiers_are_an_array and tiers_are_present
    if not tiers_are_valid:
        raise ModelValidationError("manifest.tiers must be a non-empty array")

    tiers: list[str] = []
    tier_names: set[str] = set()
    for tier_index, raw_tier in enumerate(raw_tiers):
        tier = _require_string_value(
            raw_tier,
            f"manifest.tiers[{tier_index}]",
        )
        if tier in tier_names:
            raise ModelValidationError(f"Duplicate model tier in manifest: {tier}")
        tier_names.add(tier)
        tiers.append(tier)

    entries = root.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelValidationError("manifest.models must be a non-empty array")

    models: list[ModelSpec] = []
    ids: set[str] = set()
    names: set[str] = set()
    for index, raw_entry in enumerate(entries):
        context = f"manifest.models[{index}]"
        entry = _require_object(raw_entry, context)

        model_id = _require_string(entry, "id", context)
        tier = _require_string(entry, "tier", context)
        name = _require_string(entry, "name", context)
        repository = _require_string(entry, "repository", context)
        revision = _require_string(entry, "revision", context)
        filename = _require_string(entry, "filename", context)
        sha256 = _require_string(entry, "sha256", context)
        size_bytes = entry.get("size_bytes")
        raw_modes = _require_object(entry.get("test-modes", {}), f"{context}.test-modes")
        test_modes = {}
        supported_modes = {
            "lemonade-benchmark": {"default", "mtp:on", "mtp:off"},
            "perplexity": {"default"},
            "long-context-benchmark": {"default"},
        }

        def valid_mode(check: str, mode: str) -> bool:
            if mode in supported_modes[check]:
                return True
            return (
                check == "long-context-benchmark"
                and mode.isdecimal()
                and int(mode) > 0
            )

        for check, modes in raw_modes.items():
            if check not in supported_modes:
                raise ModelValidationError(f"{context}: unknown check {check!r}")
            if known_checks is not None:
                if check not in known_checks:
                    raise ModelValidationError(f"{context}: unknown check {check!r}")
            modes_are_list = isinstance(modes, list)
            modes_are_present = bool(modes)
            valid_mode_list = modes_are_list and modes_are_present
            if not valid_mode_list:
                raise ModelValidationError(f"{context}.test-modes.{check} must be a non-empty list")
            for mode in modes:
                if not isinstance(mode, str):
                    raise ModelValidationError(f"{context}: modes for {check} must be strings")
                if not valid_mode(check, mode):
                    raise ModelValidationError(f"{context}: unsupported mode {mode!r} for {check}")
            if len(set(modes)) != len(modes):
                raise ModelValidationError(f"{context}: duplicate modes for {check}")
            test_modes[check] = tuple(modes)
        expected_results = _require_object(entry.get("expected_results", {}), context)
        for check, backends in expected_results.items():
            if known_checks is not None:
                if check not in known_checks:
                    raise ModelValidationError(f"{context}: unknown check {check!r}")
            if check not in supported_modes:
                raise ModelValidationError(f"{context}: unknown check {check!r}")
            valid_modes = test_modes.get(check, ("default",))
            for backend, modes in _require_object(backends, context).items():
                if backend not in ("hrx", "vulkan"):
                    raise ModelValidationError(f"{context}: unknown backend {backend!r}")
                for mode, result in _require_object(modes, context).items():
                    if mode not in valid_modes:
                        raise ModelValidationError(f"{context}: unused mode {mode!r} for {check}")
                    if result not in ("pass", "fail", "report-only"):
                        raise ModelValidationError(f"{context}: invalid expectation {result!r}")

        _validate_local_name(name, "name", context)
        _validate_local_name(filename, "filename", context)
        _validate_repository(repository, context)
        if tier not in tier_names:
            raise ModelValidationError(
                f"{context}.tier references unknown tier {tier!r}"
            )
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ModelValidationError(
                f"{context}.size_bytes must be a positive integer"
            )
        if model_id in ids:
            raise ModelValidationError(
                f"Duplicate model id in manifest: {model_id}"
            )
        if name in names:
            raise ModelValidationError(
                f"Duplicate runtime model name in manifest: {name}"
            )

        ids.add(model_id)
        names.add(name)
        models.append(
            ModelSpec(
                id=model_id,
                tier=tier,
                expected_results=expected_results,
                test_modes=test_modes,
                name=name,
                repository=repository,
                revision=revision,
                filename=filename,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
    return ModelManifest(tiers=tuple(tiers), models=tuple(models))


def load_validation_spec(path: Path) -> list[ValidationCheck]:
    """Load an ordered model validation specification."""
    root = _require_object(
        _load_json(path, "model validation specification"),
        "model validation specification",
    )
    entries = root.get("checks")
    entries_are_an_array = isinstance(entries, list)
    entries_are_present = bool(entries)
    entries_are_valid = entries_are_an_array and entries_are_present
    if not entries_are_valid:
        raise ModelValidationError(
            "model validation specification must contain at least one check"
        )

    check_ids: set[str] = set()
    checks: list[ValidationCheck] = []
    managed_arguments = {
        "--batched",
        "--batch-number",
        "--models-dir",
        "--models",
    }
    for entry_index, raw_entry in enumerate(entries):
        context = f"model validation specification.checks[{entry_index}]"
        entry = _require_object(raw_entry, context)
        check_id = _require_string(entry, "id", context)
        if check_id in check_ids:
            raise ModelValidationError(
                f"Duplicate check id in specification: {check_id}"
            )
        check_ids.add(check_id)

        raw_argv = entry.get("argv")
        argv_is_an_array = isinstance(raw_argv, list)
        argv_is_present = bool(raw_argv)
        argv_is_valid = argv_is_an_array and argv_is_present
        if not argv_is_valid:
            raise ModelValidationError(
                f"{context}.argv must be a non-empty string array"
            )
        argv: list[str] = []
        for argument_index, raw_argument in enumerate(raw_argv):
            argument_context = f"{context}.argv[{argument_index}]"
            argument_is_a_string = isinstance(raw_argument, str)
            argument_is_present = bool(raw_argument)
            argument_is_valid = argument_is_a_string and argument_is_present
            if not argument_is_valid:
                raise ModelValidationError(
                    f"{argument_context} must be a non-empty string"
                )
            argument_has_nul = "\0" in raw_argument
            if argument_has_nul:
                raise ModelValidationError(
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
            raise ModelValidationError(
                f"{context}.argv must not contain batch-managed arguments: "
                f"{', '.join(sorted(present_managed))}"
            )
        checks.append(ValidationCheck(id=check_id, argv=tuple(argv)))

    return checks


def select_checks(
    checks: list[ValidationCheck],
    requested: str,
) -> list[ValidationCheck]:
    """Validate the user include list and preserve specification order."""
    try:
        ids = json.loads(requested)
    except json.JSONDecodeError as exc:
        raise ModelValidationError("--include-checks must be a JSON array of IDs") from exc
    if not isinstance(ids, list):
        raise ModelValidationError("--include-checks must be a JSON array of IDs")
    for check_id in ids:
        _require_string_value(check_id, "--include-checks entry")
    known_ids = {check.id for check in checks}
    unknown_ids = set(ids) - known_ids
    if unknown_ids:
        raise ModelValidationError(
            f"Unknown model validation check IDs: {', '.join(sorted(unknown_ids))}; "
            f"available IDs: {', '.join(sorted(known_ids))}"
        )
    selected = [check for check in checks if check.id in ids]
    if not selected:
        raise ModelValidationError("--include-checks must select at least one check")
    return selected


def select_models(
    manifest: ModelManifest,
    requested_tier: str,
    model_ids_json: str = "[]",
) -> list[ModelSpec]:
    """Explicit IDs override cumulative coverage, including select-only models."""
    try:
        requested_ids = json.loads(model_ids_json)
    except json.JSONDecodeError as exc:
        raise ModelValidationError("--model-ids must be a JSON array of model IDs") from exc
    if not isinstance(requested_ids, list):
        raise ModelValidationError("--model-ids must be a JSON array of model IDs")
    for model_id in requested_ids:
        if not isinstance(model_id, str):
            raise ModelValidationError("--model-ids entries must be strings")
    if len(set(requested_ids)) != len(requested_ids):
        raise ModelValidationError("--model-ids contains duplicate IDs")
    if requested_ids:
        known_ids = {model.id for model in manifest.models}
        unknown_ids = set(requested_ids) - known_ids
        if unknown_ids:
            raise ModelValidationError(f"Unknown model IDs: {', '.join(sorted(unknown_ids))}")
        return [model for model in manifest.models if model.id in requested_ids]
    if requested_tier == "select-only":
        raise ModelValidationError("select-only models require explicit --model-ids")
    try:
        requested_tier_index = manifest.tiers.index(requested_tier)
    except ValueError:
        available_tiers = ", ".join(manifest.tiers)
        raise ModelValidationError(
            f"Unknown model tier {requested_tier!r}; "
            f"available tiers: {available_tiers}"
        ) from None

    included_tiers = set(manifest.tiers[: requested_tier_index + 1]) - {"select-only"}
    selected = [
        model for model in manifest.models if model.tier in included_tiers
    ]
    if not selected:
        raise ModelValidationError(
            f"Model tier {requested_tier!r} does not select any models"
        )
    return selected


def calculate_batch_capacity(max_disk_gib: float, free_bytes: int) -> int:
    """Reserve 2 GiB and cap model residency by policy and current free space."""
    if not math.isfinite(max_disk_gib) or max_disk_gib <= 2:
        raise ModelValidationError(
            "--max-disk-gib must be finite and greater than 2"
        )

    maximum_bytes = int(max_disk_gib * GIB)
    if free_bytes <= RESERVED_BYTES:
        raise ModelValidationError(
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
        raise ModelValidationError(
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
        headers={"User-Agent": "hrx-check/1"},
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


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse an external Retry-After header; invalid values use local backoff."""
    if value is None:
        return None
    value = value.strip()
    is_ascii = value.isascii()
    is_decimal = value.isdecimal()
    is_delay_seconds = is_ascii and is_decimal
    try:
        if is_delay_seconds:
            delay = float(value)
            return delay if math.isfinite(delay) else None
        deadline = parsedate_to_datetime(value)
        # The obsolete asctime HTTP-date format has no explicit timezone.
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, deadline.timestamp() - time.time())
    except (ValueError, OverflowError):
        return None


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
    model_dir = models_dir / model.name
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
            delay = retry_delay * attempt
            if isinstance(exc, urllib.error.HTTPError):
                header = exc.headers.get("Retry-After") if exc.headers else None
                server_delay = _retry_after_seconds(header)
                if server_delay is not None:
                    delay = max(delay, server_delay)
                elif exc.code == 429:
                    delay = max(delay, RATE_LIMIT_RETRY_DELAY * 2 ** (attempt - 1))
                exc.close()
            if attempt < attempts:
                log(
                    f"Download attempt {attempt}/{attempts} failed for "
                    f"{model.id}: {exc}; retrying in {delay:g}s from byte zero"
                )
                if delay:
                    time.sleep(delay)

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
        raise ModelValidationError(
            f"Batch cleanup did not remove target: {batch_dir}"
        )


def cleanup_work_root(work_root: Path) -> None:
    """Remove the empty work root and confirm its disappearance."""
    work_root.rmdir()
    if work_root.exists() or work_root.is_symlink():
        raise ModelValidationError(
            f"Work-root cleanup did not remove target: {work_root}"
        )


def build_check_command(
    check: ValidationCheck,
    *,
    batch_number: int,
    models_dir: Path,
    resident: list[ModelSpec],
) -> list[str]:
    return [
        *check.argv,
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
        log("Model validation completed successfully")
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
        checks = load_validation_spec(args.validation_spec)
        manifest = load_manifest(
            args.model_manifest, known_checks={check.id for check in checks}
        )
        # Model expectations may refer to workers omitted from this run.
        checks = select_checks(checks, args.include_checks)
        log(f"Selected model validation checks: {', '.join(check.id for check in checks)}")
        selected = select_models(manifest, args.model_tier, args.model_ids)
        work_root = prepare_work_root(args.work_root)

        free_bytes = shutil.disk_usage(work_root).free
        capacity_bytes = calculate_batch_capacity(
            args.max_disk_gib,
            free_bytes,
        )
        batches = plan_batches(selected, capacity_bytes)
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
                        f"Skipping model validation for batch {batch_number}: "
                        "no models downloaded successfully"
                    )
                else:
                    for check in checks:
                        command = build_check_command(
                            check,
                            batch_number=batch_number,
                            models_dir=models_dir,
                            resident=resident,
                        )
                        log("++ " + shlex.join(command))
                        try:
                            completed = subprocess.run(command, check=False)
                        except OSError as exc:
                            log(
                                f"Check {check.id} batch "
                                f"{batch_number} failed to run: {exc}"
                            )
                            failures.append(
                                "check invocation "
                                f"(batch={batch_number}, "
                                f"check={check.id}): {exc}"
                            )
                            continue
                        log(
                            f"Check {check.id} batch {batch_number} "
                            f"exited with status {completed.returncode}"
                        )
                        if completed.returncode != 0:
                            failures.append(
                                "check exit "
                                f"(batch={batch_number}, check={check.id}): "
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
                except (ModelValidationError, OSError) as exc:
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
    except (ModelValidationError, OSError) as exc:
        failures.append(f"setup: {exc}")
    finally:
        if work_root is not None and cleanup_exact:
            try:
                cleanup_work_root(work_root)
            except (ModelValidationError, OSError) as exc:
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
        "--validation-spec",
        type=Path,
        required=True,
        help="JSON command list; relative argv paths use the current directory",
    )
    parser.add_argument(
        "--include-checks",
        default='["lemonade-benchmark","perplexity"]',
        help='JSON array of check IDs to run; default: ["lemonade-benchmark","perplexity"]',
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-tier", default="single", metavar="TIER")
    parser.add_argument(
        "--model-ids", default="[]",
        help="JSON array of exact model IDs; a non-empty list overrides --model-tier",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--max-disk-gib", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ModelValidationError, OSError) as exc:
        print(f"Model validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
