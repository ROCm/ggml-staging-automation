#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write an HRX/Vulkan Lemonade benchmark report as Markdown."""

from __future__ import annotations

import argparse
import json
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
                    "| Scenario | TTFT mean (ms) | TTFT min (ms) | "
                    "TTFT max (ms) | TPS mean | TPS min | TPS max | "
                    "VRAM peak (GB) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for scenario in result["scenarios"]:
                ttft = scenario["ttft_ms"]
                tps = scenario["tps"]
                lines.append(
                    f"| {format_table_cell(scenario['name'])} | "
                    f"{ttft['mean']:.1f} | {ttft['min']:.1f} | "
                    f"{ttft['max']:.1f} | {tps['mean']:.1f} | "
                    f"{tps['min']:.1f} | {tps['max']:.1f} | "
                    f"{scenario['vram_peak_gb']:.1f} |"
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
                    f"| Scenario | Output tokens | {left_backend_label} TPS mean | "
                    f"{right_backend_label} TPS mean | TPS parity | "
                    f"{left_backend_label} TTFT mean (ms) | "
                    f"{right_backend_label} TTFT mean (ms) | "
                    f"{left_backend_label} VRAM peak (GB) | "
                    f"{right_backend_label} VRAM peak (GB) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            previous_result = result_key

        hrx_tps = hrx_scenario["tps"]["mean"]
        vulkan_tps = vulkan_scenario["tps"]["mean"]
        lines.append(
            f"| {format_table_cell(scenario_name)} | "
            f"{hrx_scenario['output_tokens']} | {hrx_tps:.1f} | "
            f"{vulkan_tps:.1f} | {format_tps_parity(hrx_tps, vulkan_tps)} | "
            f"{hrx_scenario['ttft_ms']['mean']:.1f} | "
            f"{vulkan_scenario['ttft_ms']['mean']:.1f} | "
            f"{hrx_scenario['vram_peak_gb']:.1f} | "
            f"{vulkan_scenario['vram_peak_gb']:.1f} |"
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
