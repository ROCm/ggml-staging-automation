# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared output helpers for benchmark workers run once per model batch."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON document in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as temporary_file:
        json.dump(data, temporary_file, indent=2)
        temporary_file.write("\n")
    os.replace(temporary_path, path)


def merge_benchmark_output(
    cumulative_output: Path, batch_data: dict[str, Any]
) -> int:
    """Atomically append one trusted backend batch to cumulative output."""
    if cumulative_output.exists():
        merged_data = json.loads(cumulative_output.read_text(encoding="utf-8"))
        merged_data["models"].extend(batch_data["models"])
    else:
        merged_data = batch_data

    atomic_write_json(cumulative_output, merged_data)
    return len(batch_data["models"])


def append_batch_log(destination: Path, source: Path, batch_number: int) -> None:
    """Append one temporary log between stable batch-number delimiters."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = f"batch {batch_number}"
    with destination.open("a", encoding="utf-8") as destination_file:
        destination_file.write(f"===== BEGIN {label} =====\n")
        if source.is_file():
            with source.open(
                encoding="utf-8", errors="replace"
            ) as source_file:
                shutil.copyfileobj(source_file, destination_file)
        destination_file.write(f"\n===== END {label} =====\n")
