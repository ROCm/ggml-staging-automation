#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write an HRX/Vulkan perplexity comparison report as Markdown."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


Perplexity = dict[str, Any]
Run = dict[str, Any]
ComparisonMatch = tuple[str, Run, Run]
BASELINE_UNAVAILABLE_MESSAGE = (
    "No usable main perplexity artifact is available for comparison."
)
UNAVAILABLE_MEASUREMENT = "—"
PARTIAL_FAILURE_NOTE = (
    "`—` marks an unavailable measurement; failed runs are described in the "
    "Status column with the log file and batch to inspect. Perplexity deltas "
    "are informational and never fail the job."
)


class PerplexityReportError(RuntimeError):
    """Raised when two perplexity files cannot be compared safely."""


def load_perplexity(path: Path) -> Perplexity:
    """Load a perplexity JSON file without coupling I/O to report formatting."""
    with path.open(encoding="utf-8") as perplexity_file:
        return json.load(perplexity_file)


def format_code(value: object) -> str:
    """Format trusted metadata as an inline Markdown code span."""
    text = str(value).replace("\n", " ")
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}"


def format_table_cell(value: object) -> str:
    """Escape labels for a Markdown table cell."""
    return str(value).replace("\n", " ").replace("|", "\\|")


def run_succeeded(run: Run) -> bool:
    """Return whether one run produced a usable estimate."""
    status = run["status"]
    if status not in ("ok", "failed"):
        raise PerplexityReportError(f"Run status must be ok or failed, got {status!r}")
    return status == "ok"


def run_metric(run: Run, *fields: str) -> int | float:
    """Return one finite, non-negative measurement."""
    value: Any = run
    for field in fields:
        value = value[field]
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise PerplexityReportError(
            f"Run {'.'.join(fields)} must be a finite non-negative number"
        )
    return value


def validate_run(run: Run) -> None:
    """Validate the run fields consumed by the report."""
    run_metric(run, "duration_s")
    if run_succeeded(run):
        run_metric(run, "ppl", "value")
        run_metric(run, "ppl", "uncertainty")
        return
    error = run["error"]
    error_is_string = isinstance(error, str)
    error_is_present = error_is_string and bool(error)
    if not error_is_present:
        raise PerplexityReportError("Failed run must describe its error")


def index_runs(perplexity: Perplexity, backend_label: str) -> dict[str, Run]:
    """Index runs by model and reject ambiguous comparison identities."""
    runs: dict[str, Run] = {}
    for run in perplexity["models"]:
        validate_run(run)
        model = run["model"]
        if model in runs:
            raise PerplexityReportError(
                f"Duplicate {backend_label} model: {model!r}"
            )
        runs[model] = run
    return runs


def check_comparable(
    left: Perplexity,
    right: Perplexity,
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Refuse to compare estimates that were not measured the same way."""
    same_settings = left["settings"] == right["settings"]
    same_corpus = left["corpus"]["sha256"] == right["corpus"]["sha256"]
    if not same_settings:
        raise PerplexityReportError(
            f"{left_label} and {right_label} used different perplexity settings: "
            f"{left['settings']!r} vs {right['settings']!r}"
        )
    if not same_corpus:
        raise PerplexityReportError(
            f"{left_label} and {right_label} used different corpora"
        )


def match_runs(
    left: Perplexity,
    right: Perplexity,
    *,
    left_label: str = "HRX",
    right_label: str = "Vulkan",
) -> list[ComparisonMatch]:
    """Match two run sets by model and enforce comparison integrity."""
    check_comparable(left, right, left_label=left_label, right_label=right_label)
    left_runs = index_runs(left, left_label)
    right_runs = index_runs(right, right_label)

    missing_right = [model for model in left_runs if model not in right_runs]
    missing_left = [model for model in right_runs if model not in left_runs]
    if missing_right or missing_left:
        details = []
        if missing_right:
            details.append(f"missing from {right_label}: {missing_right!r}")
        if missing_left:
            details.append(f"missing from {left_label}: {missing_left!r}")
        raise PerplexityReportError(
            "Perplexity counterparts do not match: " + " | ".join(details)
        )
    return [(model, left_runs[model], right_runs[model]) for model in left_runs]


def format_ppl(run: Run) -> str:
    if not run_succeeded(run):
        return UNAVAILABLE_MEASUREMENT
    return f"{run['ppl']['value']:.4f} ± {run['ppl']['uncertainty']:.4f}"


def format_delta(left: Run, right: Run) -> str:
    """Express the left estimate minus the right estimate."""
    both_succeeded = run_succeeded(left) and run_succeeded(right)
    if not both_succeeded:
        return UNAVAILABLE_MEASUREMENT
    return f"{left['ppl']['value'] - right['ppl']['value']:+.4f}"


def format_ratio(left: Run, right: Run) -> str:
    """Express the left estimate as a multiple of the right estimate."""
    both_succeeded = run_succeeded(left) and run_succeeded(right)
    if not both_succeeded:
        return UNAVAILABLE_MEASUREMENT
    right_value = right["ppl"]["value"]
    if right_value == 0:
        return "N/A"
    return f"{left['ppl']['value'] / right_value:.4f}"


def format_duration(run: Run) -> str:
    return f"{run['duration_s']:.0f}"


def format_run_status(run: Run, label: str) -> str:
    """Describe one failed run with the evidence needed to inspect it."""
    location = format_code(run["log"])
    if run.get("batch") is not None:
        location += f", batch {run['batch']}"
    return f"{label} failed: {format_table_cell(run['error'])} (see {location})"


def format_pair_status(
    left: Run,
    right: Run,
    left_label: str,
    right_label: str,
) -> str:
    """Summarize failures across two backends."""
    statuses = [
        format_run_status(run, label)
        for run, label in ((left, left_label), (right, right_label))
        if not run_succeeded(run)
    ]
    return "; ".join(statuses) if statuses else "OK"


def format_settings_line(perplexity: Perplexity) -> str:
    """Describe how the estimates were measured."""
    settings = perplexity["settings"]
    corpus = perplexity["corpus"]
    extra_args = settings.get("extra_args") or []
    arguments = (
        " ".join(format_code(argument) for argument in extra_args)
        if extra_args
        else "_(none)_"
    )
    return (
        f"**Corpus:** {format_code(corpus['name'])} "
        f"(sha256 {format_code(corpus['sha256'][:12])}) · "
        f"**Chunks:** {format_code(settings['chunks'])} × "
        f"{format_code(settings['ctx'])} tokens · "
        f"**Batch:** {format_code(settings['batch'])} · "
        f"**Extra llama-perplexity arguments:** {arguments}"
    )


def format_comparison_table(
    matches: Sequence[ComparisonMatch],
    settings_line: str,
    *,
    title: str = "Perplexity HRX/Vulkan comparison",
    left_label: str = "HRX",
    right_label: str = "Vulkan",
) -> str:
    """Format already-matched runs without performing I/O or validation."""
    lines = [
        f"## {title}",
        "",
        settings_line,
        "",
        f"Δ PPL is {left_label} PPL minus {right_label} PPL; ratio is "
        f"{left_label} PPL divided by {right_label} PPL.",
        "",
        f"| Model | Status | {left_label} PPL | {right_label} PPL | Δ PPL | "
        f"Ratio | {left_label} time (s) | {right_label} time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, left_run, right_run in matches:
        lines.append(
            f"| {format_code(model)} | "
            f"{format_pair_status(left_run, right_run, left_label, right_label)} | "
            f"{format_ppl(left_run)} | "
            f"{format_ppl(right_run)} | "
            f"{format_delta(left_run, right_run)} | "
            f"{format_ratio(left_run, right_run)} | "
            f"{format_duration(left_run)} | "
            f"{format_duration(right_run)} |"
        )
    return "\n".join(lines).rstrip()


def format_main_comparisons(
    current_hrx: Perplexity,
    current_vulkan: Perplexity,
    main_hrx: Perplexity,
    main_vulkan: Perplexity,
    baseline_run_url: str | None = None,
) -> str:
    """Format both current-versus-main comparisons atomically."""
    hrx_matches = match_runs(
        current_hrx,
        main_hrx,
        left_label="Current HRX",
        right_label="Main HRX",
    )
    vulkan_matches = match_runs(
        current_vulkan,
        main_vulkan,
        left_label="Current Vulkan",
        right_label="Main Vulkan",
    )

    sections = []
    if baseline_run_url:
        sections.append(
            f"Main perplexity baseline: [CI run on `main`]({baseline_run_url})"
        )
    sections.extend(
        (
            format_comparison_table(
                hrx_matches,
                format_settings_line(current_hrx),
                title="Perplexity current HRX/main HRX comparison",
                left_label="Current HRX",
                right_label="Main HRX",
            ),
            format_comparison_table(
                vulkan_matches,
                format_settings_line(current_vulkan),
                title="Perplexity current Vulkan/main Vulkan comparison",
                left_label="Current Vulkan",
                right_label="Main Vulkan",
            ),
        )
    )
    return "\n\n".join(sections)


def format_report(hrx: Perplexity, vulkan: Perplexity) -> str:
    """Match perplexity data, then format the comparison section."""
    matches = match_runs(hrx, vulkan)
    return "\n\n".join(
        (
            PARTIAL_FAILURE_NOTE,
            format_comparison_table(matches, format_settings_line(hrx)),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hrx_perplexity", type=Path)
    parser.add_argument("vulkan_perplexity", type=Path)
    parser.add_argument("--baseline-hrx-perplexity", type=Path)
    parser.add_argument("--baseline-vulkan-perplexity", type=Path)
    parser.add_argument("--baseline-run-url")
    args = parser.parse_args()

    try:
        hrx = load_perplexity(args.hrx_perplexity)
        vulkan = load_perplexity(args.vulkan_perplexity)
        report = format_report(hrx, vulkan)
    except (
        OSError,
        json.JSONDecodeError,
        PerplexityReportError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Could not write perplexity report: {exc}", file=sys.stderr)
        return 1

    baseline_paths = (
        args.baseline_hrx_perplexity,
        args.baseline_vulkan_perplexity,
    )
    if all(path is not None for path in baseline_paths):
        try:
            main_hrx = load_perplexity(args.baseline_hrx_perplexity)
            main_vulkan = load_perplexity(args.baseline_vulkan_perplexity)
            main_comparisons = format_main_comparisons(
                hrx,
                vulkan,
                main_hrx,
                main_vulkan,
                args.baseline_run_url,
            )
        except (
            OSError,
            json.JSONDecodeError,
            PerplexityReportError,
            IndexError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"Could not compare with main perplexity: {exc}", file=sys.stderr)
            report = f"{report}\n\n{BASELINE_UNAVAILABLE_MESSAGE}"
        else:
            report = f"{report}\n\n{main_comparisons}"
    else:
        if any(path is not None for path in baseline_paths):
            print(
                "Could not compare with main perplexity: both baseline perplexity "
                "paths are required",
                file=sys.stderr,
            )
        report = f"{report}\n\n{BASELINE_UNAVAILABLE_MESSAGE}"

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
