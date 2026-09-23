#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write the Lemonade HRX/Vulkan throughput report as Markdown.

CI runs the release benchmark through Lemonade against HRX and Vulkan,
leaving ``benchmark-hrx.json`` and ``benchmark-vulkan.json`` (produced by
``run_lemonade_benchmark.py``). This script turns the pair into the GitHub
step summary so a reviewer can compare generation throughput across models
without opening the artifacts. Collection decides which scenarios run; the
report discovers every scenario present in either artifact, with no fixed
list of scenarios or models and no aggregation across scenarios.

Terms:

- A *scenario* is one prompt/generation setting inside a *result*, which is one
  model run with a given recipe, context size, and backend arguments. Scenarios
  are matched across backends by their *comparison key* ``(model, recipe,
  ctx_size, backend_args, scenario name)``; the backend itself is deliberately
  not part of the key, since it is exactly what differs between the two files.
- A *missing scenario* has no entry for that key on one backend. An entry with
  ``all_runs_failed`` has no successful sample. Both have unavailable TPS,
  while a *partial failure* has ``failed_runs > 0`` and still reports statistics
  over the runs that succeeded.
- *Percentage* means HRX mean TPS divided by Vulkan mean TPS, multiplied by
  100. Thus 100% is parity, and 80% means HRX achieved 80% of Vulkan's rate.

The artifact shape, reduced to the fields this report reads (``#`` marks the
comparison key)::

    {"models": [
      {"model": "extra.example-model",                  # key
       "results": [
         {"recipe": "llamacpp",                         # key
          "ctx_size": 4096, "backend_args": "",          # key, key
          "scenarios": [
            {"name": "example-scenario",                # key
             "failed_runs": 0, "tps": {"mean": 42.0}},
            {"name": "another-scenario",                # key
             "failed_runs": 3, "all_runs_failed": true}]}]}]}

``backend_args`` defaults to an empty string and ``all_runs_failed`` to false
when omitted. Other artifact fields, such as output-token counts, TTFT, and
memory measurements, are not consumed. Different output-token counts do not
prevent a TPS comparison: the report compares the supplied rates rather than
requiring generation to stop at the same token on both backends.

The comparison includes the union of both indexes so a missing model or
scenario on one backend cannot silently remove the other backend's result.
Within a scenario, each distinct recipe/context/arguments combination gets
its own table; configurations are never averaged or paired across mismatched
keys. Duplicate keys within an artifact are rejected because they would make
that pairing ambiguous. Removing ``extra.`` affects display names only;
comparison identities retain the full model names.

Output starts with ``Lemonade benchmarks``, the optional runner name, and the
percentage definition. Each discovered scenario then gets a section named
exactly as in the artifact, with models as rows and Vulkan tok/s, HRX tok/s,
and HRX/Vulkan percentage as columns. Scenario and model order follow first
appearance in HRX, followed by entries present only in Vulkan. Configuration
labels appear when a scenario requires multiple tables. Empty inputs produce
a no-measurements note; historical comparisons are not included.

An unavailable TPS renders as ``—``. The percentage also renders as ``—``
when either rate is unavailable or Vulkan's rate is zero. Notes below each
table identify missing scenarios and reported failed runs; partial-failure
notes explain that the mean includes successful samples only. These are
benchmark outcomes to display, not malformed-input errors that fail a report.

The command line and exit-code contract live in ``benchmark_report`` and are
shared with the perplexity report; this file supplies ``kind="benchmark"``
and ``format_report``. ``main`` passes GitHub Actions' ``RUNNER_NAME`` into the
formatter so machine-dependent throughput differences have visible context.
Local rendering without that environment variable omits the runner line.

``index_scenarios`` validates the consumed measurement fields at the artifact
boundary. Counts must be non-negative integers, failure flags must be Boolean,
and usable mean TPS must be finite and non-negative. Exact numeric type checks
exclude Booleans, and explicit finiteness checks exclude NaN and infinity.
Measurements marked ``all_runs_failed`` do not require a TPS field. The shared
CLI handles validation errors before printing any report, so malformed input
cannot leave a partial summary.

``format_report`` validates and formats already-loaded dictionaries without
filesystem or environment access; callers can exercise it in memory and pass
an optional runner name. Downstream rendering consumes those validated
measurements without repeating validation.
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
