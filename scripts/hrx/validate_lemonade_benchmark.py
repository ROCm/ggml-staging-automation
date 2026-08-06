#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate that a Lemonade benchmark reported only successful scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark does not match the required result schema."""


def require_nonempty_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{location} must be a nonempty array")
    return value


def validate_benchmark(data: object) -> int:
    if not isinstance(data, dict):
        raise BenchmarkValidationError("benchmark root must be an object")

    models = require_nonempty_list(data.get("models"), "models")
    scenario_count = 0
    for model_index, model in enumerate(models):
        model_location = f"models[{model_index}]"
        if not isinstance(model, dict):
            raise BenchmarkValidationError(f"{model_location} must be an object")

        results = require_nonempty_list(
            model.get("results"), f"{model_location}.results"
        )
        for result_index, result in enumerate(results):
            result_location = f"{model_location}.results[{result_index}]"
            if not isinstance(result, dict):
                raise BenchmarkValidationError(f"{result_location} must be an object")

            scenarios = require_nonempty_list(
                result.get("scenarios"), f"{result_location}.scenarios"
            )
            for scenario_index, scenario in enumerate(scenarios):
                scenario_location = (
                    f"{result_location}.scenarios[{scenario_index}]"
                )
                if not isinstance(scenario, dict):
                    raise BenchmarkValidationError(
                        f"{scenario_location} must be an object"
                    )

                name = scenario.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise BenchmarkValidationError(
                        f"{scenario_location}.name must be a nonempty string"
                    )

                failed_runs = scenario.get("failed_runs")
                if isinstance(failed_runs, bool) or not isinstance(failed_runs, int):
                    raise BenchmarkValidationError(
                        f"{scenario_location}.failed_runs must be an integer"
                    )
                if failed_runs != 0:
                    raise BenchmarkValidationError(
                        f"{scenario_location} ({name!r}) reported "
                        f"{failed_runs} failed run(s)"
                    )
                if scenario.get("all_runs_failed") is True:
                    raise BenchmarkValidationError(
                        f"{scenario_location} ({name!r}) reports that all runs failed"
                    )
                scenario_count += 1

    return scenario_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", type=Path)
    args = parser.parse_args()

    try:
        with args.benchmark_json.open("r", encoding="utf-8") as benchmark_file:
            data = json.load(benchmark_file)
        scenario_count = validate_benchmark(data)
    except (OSError, json.JSONDecodeError, BenchmarkValidationError) as exc:
        print(f"Benchmark validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated {scenario_count} successful Lemonade benchmark scenario(s) "
        f"in {args.benchmark_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
