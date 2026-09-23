# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared CLI, error handling, and Markdown helpers for CI benchmark reports.

Two scripts turn CI benchmark artifacts into Markdown for the GitHub step
summary: ``write_lemonade_benchmark_report.py`` (Lemonade throughput) and
``write_perplexity_report.py`` (llama-perplexity). Each compares an HRX artifact
against a Vulkan artifact from the same run. The scripts differ in what a
measurement is and how tables are arranged; this module owns their common
command-line and error-handling contract and Markdown helpers. Each report
owns its measurement matching rules. Historical artifacts are no longer loaded
by either report.

Terms:

- *Current pair*: the HRX and Vulkan artifacts from the same CI run.
- *Kind*: the noun the CLI uses for the artifact, ``benchmark`` or
  ``perplexity``. It appears in positional argument names and diagnostics.

The CLI (``run_report_cli``) is invoked by each script's entry point. The
workflow supplies the two current artifacts and appends stdout to its summary::

    python3 scripts/hrx/benchmark/write_lemonade_benchmark_report.py \\
        benchmark-hrx.json benchmark-vulkan.json >> "$GITHUB_STEP_SUMMARY"

The current pair must load and format successfully before anything is printed.
On success, the complete report goes to stdout and the exit status is 0. An
input or comparison error produces a diagnostic on stderr, exit status 1, and
nothing on stdout: a misleading or partial summary is worse than a missing
one. A valid artifact describing a failed benchmark is distinct from malformed
input; the formatter renders that failure without failing the report itself.

Artifact JSON crosses the report's input boundary. ``load_json`` owns reading
and JSON decoding; each script's formatter owns measurement validation and
comparability, because those depend on the kind of benchmark. The shared CLI
wraps both loading and formatting in ``REPORT_INPUT_ERRORS``, turning problems
such as missing fields, invalid metric types, or unreadable files into the
same diagnostic contract. Scripts raise ``ReportError`` for the semantic
problems they detect themselves. Once validation establishes the indexes,
matching and presentation consume them without repeating those checks.
"""

from __future__ import annotations

import argparse
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
