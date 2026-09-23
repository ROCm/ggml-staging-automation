# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared CLI, matching, and Markdown helpers for CI benchmark reports.

The Lemonade and perplexity scripts render the current HRX/Vulkan JSON pair
into a GitHub step summary. Each formatter owns measurement validation and
presentation; this module loads the artifacts and catches input errors before
anything is printed. Malformed or unreadable input returns status 1 with a
stderr diagnostic; a complete report goes to stdout with status 0.

``match_indexed`` pairs keys present in both insertion-ordered indexes, in
left order. Perplexity uses this intersection; Lemonade retains the union so
models missing from one backend remain visible. Historical artifacts are not
loaded by either report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar


UNAVAILABLE_MEASUREMENT = "—"

K = TypeVar("K")
V = TypeVar("V")


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


def match_indexed(
    left: Mapping[K, V],
    right: Mapping[K, V],
) -> list[tuple[K, V, V]]:
    """Pair every key present in both indexes, in left order."""
    return [(key, left[key], right[key]) for key in left if key in right]


def run_report_cli(
    *,
    kind: str,
    report_label: str,
    description: str | None,
    format_report: Callable[[dict[str, Any], dict[str, Any]], str],
) -> int:
    """Parse the shared command line, build the report, return the exit code."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(f"hrx_{kind}", type=Path)
    parser.add_argument(f"vulkan_{kind}", type=Path)
    args = parser.parse_args()

    try:
        hrx = load_json(getattr(args, f"hrx_{kind}"))
        vulkan = load_json(getattr(args, f"vulkan_{kind}"))
        report = format_report(hrx, vulkan)
    except REPORT_INPUT_ERRORS as exc:
        print(f"Could not write {report_label} report: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0
