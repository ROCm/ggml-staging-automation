#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write an HRX/Vulkan Lemonade benchmark report as Markdown."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


Benchmark = dict[str, Any]
ComparisonKey = tuple[str, str, int, str, str]
ComparisonMatch = tuple[ComparisonKey, Benchmark, Benchmark]
BASELINE_UNAVAILABLE_MESSAGE = (
    "No usable main benchmark artifact is available for comparison."
)
UNAVAILABLE_MEASUREMENT = "—"
PARTIAL_FAILURE_NOTE = (
    "Statistics shown for partial failures include successful runs only; "
    "failed runs are excluded. `—` marks an unavailable measurement."
)


class BenchmarkReportError(RuntimeError):
    """Raised when two benchmark files cannot be compared safely."""


def load_benchmark(path: Path) -> Benchmark:
    """Load a benchmark JSON file without coupling I/O to report formatting."""
    with path.open(encoding="utf-8") as benchmark_file:
        return json.load(benchmark_file)


def iter_scenarios(
    benchmark: Benchmark,
) -> Iterator[tuple[ComparisonKey, Benchmark]]:
    """Yield comparison keys and scenarios in benchmark order."""
    for model in benchmark["models"]:
        model_name = model["model"]
        for result in model["results"]:
            result_key = (
                model_name,
                result["recipe"],
                result["ctx_size"],
                result.get("backend_args", ""),
            )
            for scenario in result["scenarios"]:
                yield (*result_key, scenario["name"]), scenario


def describe_key(key: ComparisonKey) -> str:
    model, recipe, context, backend_args, scenario = key
    arguments = backend_args or "<none>"
    return (
        f"model={model!r}, recipe={recipe!r}, context={context}, "
        f"backend_args={arguments!r}, scenario={scenario!r}"
    )


def index_scenarios(
    benchmark: Benchmark, backend_name: str
) -> dict[ComparisonKey, Benchmark]:
    """Index scenarios and reject ambiguous comparison identities."""
    scenarios: dict[ComparisonKey, Benchmark] = {}
    for key, scenario in iter_scenarios(benchmark):
        validate_scenario(scenario)
        if key in scenarios:
            raise BenchmarkReportError(
                f"Duplicate {backend_name} comparison key: {describe_key(key)}"
            )
        scenarios[key] = scenario
    return scenarios


def match_scenarios(
    left_benchmark: Benchmark,
    right_benchmark: Benchmark,
    *,
    left_backend_label: str = "HRX",
    right_backend_label: str = "Vulkan",
) -> list[ComparisonMatch]:
    """Match two benchmark result sets and enforce comparison integrity."""
    left_scenarios = index_scenarios(left_benchmark, left_backend_label)
    right_scenarios = index_scenarios(right_benchmark, right_backend_label)

    missing_right = [key for key in left_scenarios if key not in right_scenarios]
    missing_left = [key for key in right_scenarios if key not in left_scenarios]
    if missing_right or missing_left:
        details = []
        if missing_right:
            details.append(
                f"missing from {right_backend_label}: "
                + "; ".join(describe_key(key) for key in missing_right)
            )
        if missing_left:
            details.append(
                f"missing from {left_backend_label}: "
                + "; ".join(describe_key(key) for key in missing_left)
            )
        raise BenchmarkReportError(
            "Benchmark counterparts do not match: " + " | ".join(details)
        )

    matches: list[ComparisonMatch] = []
    for key, left_scenario in left_scenarios.items():
        right_scenario = right_scenarios[key]
        left_has_measurements = scenario_has_measurements(left_scenario)
        right_has_measurements = scenario_has_measurements(right_scenario)
        if left_has_measurements and right_has_measurements:
            left_tokens = left_scenario["output_tokens"]
            right_tokens = right_scenario["output_tokens"]
            if left_tokens != right_tokens:
                raise BenchmarkReportError(
                    f"Output-token count mismatch for {describe_key(key)}: "
                    f"{left_backend_label}={left_tokens}, "
                    f"{right_backend_label}={right_tokens}"
                )
        matches.append((key, left_scenario, right_scenario))
    return matches


def format_code(value: object) -> str:
    """Format trusted benchmark metadata as an inline Markdown code span."""
    text = str(value).replace("\n", " ")
    fence = "``" if "`" in text else "`"
    return f"{fence}{text}{fence}"


def format_table_cell(value: object) -> str:
    """Escape benchmark labels for a Markdown table cell."""
    return str(value).replace("\n", " ").replace("|", "\\|")


def scenario_failed_runs(scenario: Benchmark) -> int:
    """Return a validated failed-run count from one Lemonade scenario."""
    failed_runs = scenario["failed_runs"]
    if type(failed_runs) is not int or failed_runs < 0:
        raise BenchmarkReportError(
            "Scenario failed_runs must be a non-negative integer"
        )
    return failed_runs


def scenario_has_measurements(scenario: Benchmark) -> bool:
    """Return whether Lemonade retained at least one successful sample."""
    all_runs_failed = scenario.get("all_runs_failed", False)
    if type(all_runs_failed) is not bool:
        raise BenchmarkReportError(
            "Scenario all_runs_failed must be a Boolean when present"
        )
    return not all_runs_failed


def scenario_output_tokens(scenario: Benchmark) -> int:
    """Return a validated output-token count."""
    output_tokens = scenario["output_tokens"]
    if type(output_tokens) is not int or output_tokens < 0:
        raise BenchmarkReportError(
            "Scenario output_tokens must be a non-negative integer"
        )
    return output_tokens


def scenario_metric(
    scenario: Benchmark,
    field: str,
    statistic: str | None = None,
    *,
    optional: bool = False,
) -> int | float | None:
    """Return one finite, non-negative measurement."""
    if optional and field not in scenario:
        return None
    value = scenario[field]
    if statistic is not None:
        value = value[statistic]
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
    ):
        description = f"{field}.{statistic}" if statistic else field
        raise BenchmarkReportError(
            f"Scenario {description} must be a finite non-negative number"
        )
    return value


def validate_scenario(scenario: Benchmark) -> None:
    """Validate the scenario fields consumed by the report."""
    scenario_failed_runs(scenario)
    output_tokens = scenario_output_tokens(scenario)
    if not scenario_has_measurements(scenario):
        if output_tokens != 0:
            raise BenchmarkReportError(
                "Scenario without measurements must report zero output_tokens"
            )
        unexpected_fields = [
            field
            for field in ("ttft_ms", "tps", "vram_peak_gb")
            if field in scenario
        ]
        if unexpected_fields:
            raise BenchmarkReportError(
                "Scenario without measurements contains measurement fields: "
                + ", ".join(unexpected_fields)
            )
        return

    for field in ("ttft_ms", "tps"):
        for statistic in ("mean", "min", "max"):
            scenario_metric(scenario, field, statistic)
    scenario_metric(scenario, "vram_peak_gb", optional=True)


def format_scenario_status(scenario: Benchmark, backend_label: str) -> str:
    """Format one backend's status for a scenario row."""
    failed_runs = scenario_failed_runs(scenario)
    if not scenario_has_measurements(scenario):
        return f"{backend_label} missing ({failed_runs} failed)"
    if failed_runs:
        return f"{failed_runs} failed"
    return "OK"


def format_pair_status(
    left_scenario: Benchmark,
    right_scenario: Benchmark,
    left_backend_label: str,
    right_backend_label: str,
) -> str:
    """Summarize failures and missing measurements across two backends."""
    left_failed_runs = scenario_failed_runs(left_scenario)
    right_failed_runs = scenario_failed_runs(right_scenario)
    left_has_measurements = scenario_has_measurements(left_scenario)
    right_has_measurements = scenario_has_measurements(right_scenario)

    if left_has_measurements and right_has_measurements:
        failed_runs = left_failed_runs + right_failed_runs
        return f"{failed_runs} failed" if failed_runs else "OK"

    statuses = []
    for label, failed_runs, has_measurements in (
        (
            left_backend_label,
            left_failed_runs,
            left_has_measurements,
        ),
        (
            right_backend_label,
            right_failed_runs,
            right_has_measurements,
        ),
    ):
        if not has_measurements:
            statuses.append(f"{label} missing ({failed_runs} failed)")
        elif failed_runs:
            statuses.append(f"{label}: {failed_runs} failed")
    return "; ".join(statuses)


def format_metric(
    scenario: Benchmark,
    field: str,
    statistic: str | None = None,
    *,
    optional: bool = False,
) -> str:
    """Format one measurement, preserving malformed usable data as fatal."""
    if not scenario_has_measurements(scenario):
        return UNAVAILABLE_MEASUREMENT
    value = scenario_metric(
        scenario,
        field,
        statistic,
        optional=optional,
    )
    if value is None:
        return UNAVAILABLE_MEASUREMENT
    return f"{value:.1f}"


def format_output_tokens(
    left_scenario: Benchmark, right_scenario: Benchmark
) -> str:
    """Use the token count from either available side of a comparison."""
    if scenario_has_measurements(left_scenario):
        return str(scenario_output_tokens(left_scenario))
    if scenario_has_measurements(right_scenario):
        return str(scenario_output_tokens(right_scenario))
    return UNAVAILABLE_MEASUREMENT


def format_backend_table(benchmark: Benchmark, backend_name: str) -> str:
    """Format one backend benchmark without performing I/O or validation."""
    lines = [f"## Lemonade {backend_name} benchmark results", ""]

    for model in benchmark["models"]:
        lines.extend([f"### {format_code(model['model'])}", ""])
        for result in model["results"]:
            backend = f"{result['recipe']}/{result['backend']}"
            details = (
                f"**Backend:** {format_code(backend)} · "
                f"**Context:** {format_code(result['ctx_size'])} tokens"
            )
            backend_args = result.get("backend_args", "")
            if backend_args:
                details += f" · **Arguments:** {format_code(backend_args)}"
            lines.extend(
                [
                    details,
                    "",
                    "| Scenario | Status | TTFT mean (ms) | TTFT min (ms) | "
                    "TTFT max (ms) | TPS mean | TPS min | TPS max | "
                    "VRAM peak (GB) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
                    "---: | ---: |",
                ]
            )
            for scenario in result["scenarios"]:
                lines.append(
                    f"| {format_table_cell(scenario['name'])} | "
                    f"{format_scenario_status(scenario, backend_name)} | "
                    f"{format_metric(scenario, 'ttft_ms', 'mean')} | "
                    f"{format_metric(scenario, 'ttft_ms', 'min')} | "
                    f"{format_metric(scenario, 'ttft_ms', 'max')} | "
                    f"{format_metric(scenario, 'tps', 'mean')} | "
                    f"{format_metric(scenario, 'tps', 'min')} | "
                    f"{format_metric(scenario, 'tps', 'max')} | "
                    f"{format_metric(scenario, 'vram_peak_gb', optional=True)} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip()


def format_tps_parity(left_tps: float, right_tps: float) -> str:
    """Express the left mean TPS as a percentage of the right mean TPS."""
    if right_tps == 0:
        return "N/A"
    return f"{left_tps / right_tps * 100:.1f}%"


def format_comparison_table(
    matches: Sequence[ComparisonMatch],
    *,
    title: str = "Lemonade HRX/Vulkan comparison",
    left_backend_label: str = "HRX",
    right_backend_label: str = "Vulkan",
) -> str:
    """Format already-matched results without performing I/O or validation."""
    lines = [
        f"## {title}",
        "",
        f"TPS parity is {left_backend_label} mean TPS as a percentage of "
        f"{right_backend_label} mean TPS.",
        "",
    ]
    previous_model: str | None = None
    previous_result: tuple[str, str, int, str] | None = None

    for key, hrx_scenario, vulkan_scenario in matches:
        model, recipe, context, backend_args, scenario_name = key
        result_key = (model, recipe, context, backend_args)
        if result_key != previous_result:
            if previous_result is not None:
                lines.append("")
            if model != previous_model:
                lines.extend([f"### {format_code(model)}", ""])
                previous_model = model

            arguments = format_code(backend_args) if backend_args else "_(none)_"
            lines.extend(
                [
                    f"**Recipe:** {format_code(recipe)} · "
                    f"**Context:** {format_code(context)} tokens · "
                    f"**Backend arguments:** {arguments}",
                    "",
                    f"| Scenario | Status | Output tokens | "
                    f"{left_backend_label} TPS mean | "
                    f"{right_backend_label} TPS mean | TPS parity | "
                    f"{left_backend_label} TTFT mean (ms) | "
                    f"{right_backend_label} TTFT mean (ms) | "
                    f"{left_backend_label} VRAM peak (GB) | "
                    f"{right_backend_label} VRAM peak (GB) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | "
                    "---: | ---: | ---: |",
                ]
            )
            previous_result = result_key

        both_have_measurements = scenario_has_measurements(
            hrx_scenario
        ) and scenario_has_measurements(vulkan_scenario)
        parity = "N/A"
        if both_have_measurements:
            parity = format_tps_parity(
                hrx_scenario["tps"]["mean"],
                vulkan_scenario["tps"]["mean"],
            )
        status = format_pair_status(
            hrx_scenario,
            vulkan_scenario,
            left_backend_label,
            right_backend_label,
        )
        lines.append(
            f"| {format_table_cell(scenario_name)} | "
            f"{status} | "
            f"{format_output_tokens(hrx_scenario, vulkan_scenario)} | "
            f"{format_metric(hrx_scenario, 'tps', 'mean')} | "
            f"{format_metric(vulkan_scenario, 'tps', 'mean')} | {parity} | "
            f"{format_metric(hrx_scenario, 'ttft_ms', 'mean')} | "
            f"{format_metric(vulkan_scenario, 'ttft_ms', 'mean')} | "
            f"{format_metric(hrx_scenario, 'vram_peak_gb', optional=True)} | "
            f"{format_metric(vulkan_scenario, 'vram_peak_gb', optional=True)} |"
        )

    return "\n".join(lines).rstrip()


def format_main_comparisons(
    current_hrx_benchmark: Benchmark,
    current_vulkan_benchmark: Benchmark,
    main_hrx_benchmark: Benchmark,
    main_vulkan_benchmark: Benchmark,
    baseline_run_url: str | None = None,
) -> str:
    """Format both current-versus-main comparisons atomically."""
    hrx_matches = match_scenarios(
        current_hrx_benchmark,
        main_hrx_benchmark,
        left_backend_label="Current HRX",
        right_backend_label="Main HRX",
    )
    vulkan_matches = match_scenarios(
        current_vulkan_benchmark,
        main_vulkan_benchmark,
        left_backend_label="Current Vulkan",
        right_backend_label="Main Vulkan",
    )

    sections = []
    if baseline_run_url:
        sections.append(
            f"Main benchmark baseline: [CI run on `main`]({baseline_run_url})"
        )
    sections.extend(
        (
            format_comparison_table(
                hrx_matches,
                title="Lemonade current HRX/main HRX comparison",
                left_backend_label="Current HRX",
                right_backend_label="Main HRX",
            ),
            format_comparison_table(
                vulkan_matches,
                title="Lemonade current Vulkan/main Vulkan comparison",
                left_backend_label="Current Vulkan",
                right_backend_label="Main Vulkan",
            ),
        )
    )
    return "\n\n".join(sections)


def format_report(hrx_benchmark: Benchmark, vulkan_benchmark: Benchmark) -> str:
    """Match benchmark data, then format all three report sections."""
    matches = match_scenarios(hrx_benchmark, vulkan_benchmark)
    return "\n\n".join(
        (
            PARTIAL_FAILURE_NOTE,
            format_backend_table(hrx_benchmark, "HRX"),
            format_backend_table(vulkan_benchmark, "Vulkan"),
            format_comparison_table(matches),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hrx_benchmark", type=Path)
    parser.add_argument("vulkan_benchmark", type=Path)
    parser.add_argument("--baseline-hrx-benchmark", type=Path)
    parser.add_argument("--baseline-vulkan-benchmark", type=Path)
    parser.add_argument("--baseline-run-url")
    args = parser.parse_args()

    try:
        hrx_benchmark = load_benchmark(args.hrx_benchmark)
        vulkan_benchmark = load_benchmark(args.vulkan_benchmark)
        report = format_report(hrx_benchmark, vulkan_benchmark)
    except (
        OSError,
        json.JSONDecodeError,
        BenchmarkReportError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Could not write Lemonade benchmark report: {exc}", file=sys.stderr)
        return 1

    baseline_paths = (
        args.baseline_hrx_benchmark,
        args.baseline_vulkan_benchmark,
    )
    if all(path is not None for path in baseline_paths):
        try:
            main_hrx_benchmark = load_benchmark(args.baseline_hrx_benchmark)
            main_vulkan_benchmark = load_benchmark(args.baseline_vulkan_benchmark)
            main_comparisons = format_main_comparisons(
                hrx_benchmark,
                vulkan_benchmark,
                main_hrx_benchmark,
                main_vulkan_benchmark,
                args.baseline_run_url,
            )
        except (
            OSError,
            json.JSONDecodeError,
            BenchmarkReportError,
            IndexError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"Could not compare with main benchmark: {exc}", file=sys.stderr)
            report = f"{report}\n\n{BASELINE_UNAVAILABLE_MESSAGE}"
        else:
            report = f"{report}\n\n{main_comparisons}"
    else:
        if any(path is not None for path in baseline_paths):
            print(
                "Could not compare with main benchmark: both baseline benchmark "
                "paths are required",
                file=sys.stderr,
            )
        report = f"{report}\n\n{BASELINE_UNAVAILABLE_MESSAGE}"

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
