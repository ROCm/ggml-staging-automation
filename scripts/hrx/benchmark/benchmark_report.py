# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared output, error handling, and Markdown helpers for CI benchmark reports.

Two scripts turn CI benchmark artifacts into Markdown for the GitHub step
summary: ``write_lemonade_benchmark_report.py`` (Lemonade throughput) and
``write_perplexity_report.py`` (llama-perplexity). Lemonade compares separate HRX
and Vulkan artifacts; perplexity reads one combined artifact. Each script owns
its argument parsing and report layout. This module provides JSON loading,
Markdown helpers, and ``write_report`` for their common output contract.

The caller supplies a callback that loads its inputs and builds the complete
report. Loading stays inside the callback so reading and formatting errors
receive the same handling, regardless of the number of input artifacts.
On success, the complete report goes to stdout and the exit status is 0. An
input or comparison error produces a diagnostic on stderr, exit status 1, and
nothing on stdout: a misleading or partial summary is worse than a missing
one. A valid artifact describing a failed benchmark is distinct from malformed
input; the formatter renders that failure without failing the report itself.

``load_json`` owns reading and JSON decoding. Each report owns interpretation
of its artifact, including any comparison rules. ``write_report`` catches
``REPORT_INPUT_ERRORS`` from the callback; scripts can raise ``ReportError``
for semantic problems they detect themselves.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


UNAVAILABLE_MEASUREMENT = "—"

class ReportError(RuntimeError):
    """Raised when two report inputs cannot be compared safely."""


REPORT_INPUT_ERRORS = (
    OSError,
    json.JSONDecodeError,
    ReportError,
    IndexError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact without coupling I/O to report formatting."""
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def format_code(value: object) -> str:
    """Format trusted metadata as an inline Markdown code span."""
    text = str(value).replace("\n", " ")
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}"


def format_table_cell(value: object) -> str:
    """Escape labels for a Markdown table cell."""
    return str(value).replace("\n", " ").replace("|", "\\|")


def write_report(
    report_label: str,
    build_report: Callable[[], str],
) -> int:
    """Build the complete report before emitting anything to stdout."""
    try:
        report = build_report()
    except REPORT_INPUT_ERRORS as exc:
        print(f"Could not write {report_label} report: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0
