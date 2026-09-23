#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Render current Lemonade throughput as per-scenario tables for CI.

The HRX and Vulkan JSON artifacts contain per-model scenario measurements.
This report discovers every scenario and model in either artifact, with
models as rows and mean tok/s plus HRX/Vulkan percent as columns. Scenarios
retain their artifact names and appear in HRX order, followed by Vulkan-only
scenarios. Display names omit the ``extra.`` model prefix; matching uses full
names. It does not load historical artifacts or aggregate scenarios.

The CLI includes GitHub Actions' ``RUNNER_NAME`` beneath the title when
available. Local rendering without that environment variable omits the line.

Measurements are paired by model, recipe, context size, backend arguments,
and scenario. Different configurations get separate tables rather than being
silently combined. The union of both backends' models remains visible;
missing or failed measurements render as an em dash, and partial failures
are noted below the table. A zero Vulkan rate has no defined percentage.

The shared CLI loads the current pair and reports malformed input on stderr
without emitting a partial report. This module validates the fields it
consumes at the artifact boundary.
"""

from __future__ import annotations

import math
import os
from functools import partial
from typing import Any

from benchmark_report import (
    UNAVAILABLE_MEASUREMENT,
    ReportError,
    format_code,
    format_table_cell,
    run_report_cli,
)

Benchmark = dict[str, Any]
ComparisonKey = tuple[str, str, int, str, str]


def index_scenarios(benchmark: Benchmark) -> dict[ComparisonKey, Benchmark]:
    """Validate external measurements and establish unique identities."""
    scenarios = {}
    for model in benchmark["models"]:
        for result in model["results"]:
            for scenario in result["scenarios"]:
                name = scenario["name"]
                key = (
                    model["model"],
                    result["recipe"],
                    result["ctx_size"],
                    result.get("backend_args", ""),
                    name,
                )
                if key in scenarios:
                    raise ReportError(f"Duplicate scenario key: {key!r}")
                failed = scenario["failed_runs"]
                if type(failed) is not int:
                    raise ReportError("Scenario failed_runs must be an integer")
                if failed < 0:
                    raise ReportError("Scenario failed_runs must be non-negative")
                all_failed = scenario.get("all_runs_failed", False)
                if type(all_failed) is not bool:
                    raise ReportError("Scenario all_runs_failed must be a Boolean")
                if not all_failed:
                    rate = scenario["tps"]["mean"]
                    if type(rate) not in (int, float):
                        raise ReportError("Scenario mean TPS must be numeric")
                    finite = math.isfinite(rate)
                    nonnegative = rate >= 0
                    valid_rate = finite and nonnegative
                    if not valid_rate:
                        raise ReportError(
                            "Scenario mean TPS must be finite and non-negative"
                        )
                scenarios[key] = scenario
    return scenarios


def mean_tps(scenario: Benchmark | None) -> float | None:
    if scenario is None:
        return None
    if scenario.get("all_runs_failed", False):
        return None
    return scenario["tps"]["mean"]


def format_report(
    hrx: Benchmark, vulkan: Benchmark, *, runner_name: str | None = None
) -> str:
    """Build scenario tables from validated indexes, preserving model order."""
    indexes = {"Vulkan": index_scenarios(vulkan), "HRX": index_scenarios(hrx)}
    keys = list(dict.fromkeys([*indexes["HRX"], *indexes["Vulkan"]]))
    lines = ["# Lemonade benchmarks"]
    if runner_name:
        lines.extend(["", f"**Runner:** {format_code(runner_name)}"])
    lines.extend([
        "",
        "Percentage is HRX / Vulkan mean tok/s × 100. "
        "`—` marks an unavailable measurement or percentage.",
    ])
    scenario_names = list(dict.fromkeys(key[-1] for key in keys))
    if not scenario_names:
        lines.extend(["", "No measurements available."])
    for scenario_name in scenario_names:
        label = format_table_cell(scenario_name)
        lines.extend(["", f"## Lemonade scenario: {label}", ""])
        selected = [key for key in keys if key[-1] == scenario_name]
        configs = list(dict.fromkeys(key[1:4] for key in selected))
        for config in configs:
            model_keys = [key for key in selected if key[1:4] == config]
            # Only expose configuration metadata when needed to distinguish tables.
            if len(configs) > 1:
                recipe, context, arguments = config
                lines.extend([
                    f"**Recipe:** {format_code(recipe)} · "
                    f"**Context:** {context} · "
                    f"**Arguments:** {format_code(arguments or '<none>')}",
                    "",
                ])
            lines.append("| Model | Vulkan tok/s | HRX tok/s | HRX / Vulkan |")
            lines.append("| --- | ---: | ---: | ---: |")
            notes = []
            for key in model_keys:
                model_name = key[0].removeprefix("extra.")
                values = [format_table_cell(model_name)]
                rates = {}
                for backend, index in indexes.items():
                    scenario = index.get(key)
                    rate = mean_tps(scenario)
                    values.append(
                        UNAVAILABLE_MEASUREMENT if rate is None else f"{rate:.1f}"
                    )
                    rates[backend] = rate
                    if scenario is None:
                        notes.append(
                            f"{format_code(model_name)} {backend}: missing scenario."
                        )
                    elif scenario["failed_runs"]:
                        count = scenario["failed_runs"]
                        detail = (
                            "no successful measurements"
                            if rate is None
                            else "mean includes successful runs only"
                        )
                        notes.append(
                            f"{format_code(model_name)} {backend}: "
                            f"{count} failed runs; {detail}."
                        )
                hrx_rate = rates["HRX"]
                vulkan_rate = rates["Vulkan"]
                has_hrx = hrx_rate is not None
                has_denominator = vulkan_rate not in (None, 0)
                comparable = has_hrx and has_denominator
                values.append(
                    f"{hrx_rate / vulkan_rate * 100:.1f}%"
                    if comparable
                    else UNAVAILABLE_MEASUREMENT
                )
                lines.append("| " + " | ".join(values) + " |")
            if notes:
                lines.extend(["", "\n\n".join(notes)])
            lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    return run_report_cli(
        kind="benchmark",
        report_label="Lemonade benchmark",
        description=__doc__,
        format_report=partial(format_report, runner_name=os.environ.get("RUNNER_NAME")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
