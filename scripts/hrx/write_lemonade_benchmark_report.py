#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write the Lemonade HRX/Vulkan benchmark report as Markdown.

CI runs the release benchmark twice through Lemonade, once against the HRX
backend and once against Vulkan, and each run leaves a ``benchmark-*.json``
artifact (produced by ``run_lemonade_benchmark.py``). This script turns the pair
into the Markdown that lands in the GitHub step summary, so someone reading the
job page can see how HRX did next to Vulkan without opening the artifacts.

Terms:

- A *scenario* is one prompt/generation setting inside a *result*, which is one
  model run with a given recipe, context size, and backend arguments. Scenarios
  are matched across backends by their *comparison key* ``(model, recipe,
  ctx_size, backend_args, scenario name)``; the backend itself is deliberately
  not part of the key, since it is exactly what differs between the two files.
- A scenario is *missing* when Lemonade retained no successful sample
  (``all_runs_failed``); a *partial failure* has ``failed_runs > 0`` but still
  reports statistics over the runs that succeeded.

The artifact shape, reduced to the fields this report reads (``#`` marks the
comparison key)::

    {"models": [
      {"model": "llama-3.1-8b",                          # key
       "results": [
         {"recipe": "llamacpp", "backend": "hrx",        # recipe: key
          "ctx_size": 4096, "backend_args": "",          # key, key
          "scenarios": [
            {"name": "p128g64",                          # key
             "failed_runs": 0, "output_tokens": 64,
             "ttft_ms": {"mean": .., "min": .., "max": ..},
             "tps":     {"mean": .., "min": .., "max": ..},
             "vram_peak_gb": 4.5},                       # optional
            {"name": "p1024g64", "failed_runs": 3,
             "output_tokens": 0, "all_runs_failed": true}]}]}]}

The command line, the exit-code contract (current pair fatal, ``main``
baseline best-effort), and the matching rule all live in ``benchmark_report``
and are shared with the perplexity report; this file supplies
``kind="benchmark"`` and the section builders. Under that rule a model is
compared only when both files contain it, and then all of its scenarios must
be present on both sides. On top of it, this report requires matched
scenarios to report the same output-token count wherever both sides have
measurements: a TPS comparison over different generation lengths would be
meaningless, so a mismatch is treated like a missing counterpart (fatal for
the current pair, baseline-degrading for ``main``).

The report is printed to stdout in a fixed order: the partial-failure note, a
per-backend table for HRX then Vulkan (every model each side ran, so a model
skipped by the comparison is still visible), an HRX/Vulkan comparison, and
finally either two current-versus-main comparisons (HRX and Vulkan each
against the latest ``main`` artifact) or a one-line note that no baseline was
usable.

Validation is intentionally strict on shape (``type(x) is int`` rather than
``isinstance``) so a Boolean or NaN never masquerades as a count or a metric,
but it only covers the fields the report reads. Everything in this file is
pure formatting over already-loaded dicts, so the section builders can be
exercised in memory without touching the filesystem.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

from benchmark_report import (
    UNAVAILABLE_MEASUREMENT,
    ReportError,
    format_code,
    format_table_cell,
    match_indexed,
    run_report_cli,
)


Benchmark = dict[str, Any]
ComparisonKey = tuple[str, str, int, str, str]
ComparisonMatch = tuple[ComparisonKey, Benchmark, Benchmark]
PARTIAL_FAILURE_NOTE = (
    "Statistics shown for partial failures include successful runs only; "
    "failed runs are excluded. `—` marks an unavailable measurement."
)


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
            raise ReportError(
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
    """Match scenarios of every model both sides benchmarked."""

    def check_output_tokens(
        key: ComparisonKey,
        left_scenario: Benchmark,
        right_scenario: Benchmark,
    ) -> None:
        left_has_measurements = scenario_has_measurements(left_scenario)
        right_has_measurements = scenario_has_measurements(right_scenario)
        both_have_measurements = left_has_measurements and right_has_measurements
        if not both_have_measurements:
            return
        left_tokens = left_scenario["output_tokens"]
        right_tokens = right_scenario["output_tokens"]
        if left_tokens != right_tokens:
            raise ReportError(
                f"Output-token count mismatch for {describe_key(key)}: "
                f"{left_backend_label}={left_tokens}, "
                f"{right_backend_label}={right_tokens}"
            )

    return match_indexed(
        index_scenarios(left_benchmark, left_backend_label),
        index_scenarios(right_benchmark, right_backend_label),
        left_label=left_backend_label,
        right_label=right_backend_label,
        group=lambda key: key[0],
        describe=describe_key,
        check_pair=check_output_tokens,
    )


def scenario_failed_runs(scenario: Benchmark) -> int:
    """Return a validated failed-run count from one Lemonade scenario."""
    failed_runs = scenario["failed_runs"]
    if type(failed_runs) is not int or failed_runs < 0:
        raise ReportError(
            "Scenario failed_runs must be a non-negative integer"
        )
    return failed_runs


def scenario_has_measurements(scenario: Benchmark) -> bool:
    """Return whether Lemonade retained at least one successful sample."""
    all_runs_failed = scenario.get("all_runs_failed", False)
    if type(all_runs_failed) is not bool:
        raise ReportError(
            "Scenario all_runs_failed must be a Boolean when present"
        )
    return not all_runs_failed


def scenario_output_tokens(scenario: Benchmark) -> int:
    """Return a validated output-token count."""
    output_tokens = scenario["output_tokens"]
    if type(output_tokens) is not int or output_tokens < 0:
        raise ReportError(
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
        raise ReportError(
            f"Scenario {description} must be a finite non-negative number"
        )
    return value


def validate_scenario(scenario: Benchmark) -> None:
    """Validate the scenario fields consumed by the report."""
    scenario_failed_runs(scenario)
    output_tokens = scenario_output_tokens(scenario)
    if not scenario_has_measurements(scenario):
        if output_tokens != 0:
            raise ReportError(
                "Scenario without measurements must report zero output_tokens"
            )
        unexpected_fields = [
            field
            for field in ("ttft_ms", "tps", "vram_peak_gb")
            if field in scenario
        ]
        if unexpected_fields:
            raise ReportError(
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
    if not matches:
        lines.append(
            f"No model was benchmarked by both {left_backend_label} and "
            f"{right_backend_label}."
        )
        return "\n".join(lines)
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
    return run_report_cli(
        kind="benchmark",
        report_label="Lemonade benchmark",
        description=__doc__,
        format_report=format_report,
        format_main_comparisons=format_main_comparisons,
    )


if __name__ == "__main__":
    raise SystemExit(main())
