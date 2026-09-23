# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared CLI, matching rule, and Markdown helpers for the HRX report scripts.

Two scripts turn CI benchmark artifacts into Markdown for the GitHub step
summary: ``write_lemonade_benchmark_report.py`` (Lemonade throughput) and
``write_perplexity_report.py`` (llama-perplexity). Each compares an HRX artifact
against a Vulkan artifact from the same run. The scripts differ in what a
measurement is and how tables are arranged; this module owns their common
command-line and error-handling contract, plus reusable matching and Markdown
helpers. Historical artifacts are no longer loaded by either report.

Terms:

- *Comparison key*: the identity of one measurement inside an artifact. For
  perplexity it is the model name; for Lemonade it is
  ``(model, recipe, ctx_size, backend_args, scenario)``. The backend is never
  part of the key because it is exactly what differs between the two files.
- *Index*: an insertion-ordered ``dict`` from comparison key to the measurement
  it identifies. Each script builds its own indexes and validates entries;
  the matching helper consumes those established identities.
- *Current pair*: the HRX and Vulkan artifacts from the same CI run.
- *Kind*: the noun the CLI uses for the artifact, ``benchmark`` or
  ``perplexity``. It appears in positional argument names and diagnostics.

Matching (``match_indexed``) pairs every comparison key present in both
indexes and returns ``[(key, left_item, right_item), ...]`` in left order; a
key found on one side only is skipped, never an error. Perplexity uses this
intersection. Lemonade instead renders the union of its indexes, so scenarios
and models present on only one backend still appear with missing-value cells.
Both reports retain failed measurements in their comparisons rather than
refusing to pair them merely because one backend has no successful sample.

The two files can disagree on what they contain: a batch's HRX and Vulkan
phases run one after the other and merge per phase, so a Vulkan failure can
leave the HRX artifact with extra models. An earlier rule required identical
key sets and discarded otherwise useful comparisons on such mismatches. The
intersection helper preserves every shared key, regardless of order::

    >>> left = {("A", "p1"): 1, ("A", "p2"): 2, ("B", "p1"): 3}
    >>> right = {("A", "p1"): 10, ("C", "p1"): 30, ("A", "p2"): 20}
    >>> match_indexed(left, right)
    [(('A', 'p1'), 1, 10), (('A', 'p2'), 2, 20)]

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
