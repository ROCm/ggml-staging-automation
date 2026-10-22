#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare HRX and Vulkan for the same model, test mode, context and scenario.

Recipe names differ intentionally: HRX uses llamacpp-hrx and Vulkan llamacpp.
The worker gives expanded cases distinct names and records both outcomes.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from benchmark_report import ReportError, format_table_cell, load_json, write_report


def index_scenarios(document):
    indexed = {}
    for model in document["models"]:
        for result in model["results"]:
            for scenario in result["scenarios"]:
                key = (model["model"], result["ctx_size"], scenario["name"])
                if key in indexed:
                    raise ReportError(f"Duplicate comparison {key}")
                failed = scenario["failed_runs"]
                integer_count = type(failed) is int
                if not integer_count:
                    raise ReportError("Non-integer failed-run count")
                if failed < 0:
                    raise ReportError("Invalid failed-run count")
                rate = None
                if not scenario.get("all_runs_failed", False):
                    rate = scenario["tps"]["mean"]
                    numeric = type(rate) in (int, float)
                    if not numeric:
                        raise ReportError("Non-numeric throughput")
                    finite_rate = math.isfinite(rate)
                    nonnegative_rate = rate >= 0
                    valid_rate = finite_rate and nonnegative_rate
                    if not valid_rate:
                        raise ReportError("Invalid throughput")
                indexed[key] = (model, rate)
    return indexed


def format_report(hrx, vulkan, *, runner_name=None):
    h = index_scenarios(hrx)
    v = index_scenarios(vulkan)
    if h.keys() != v.keys():
        raise ReportError("HRX/Vulkan cases differ; comparison is incomplete")
    lines = ["# Lemonade benchmarks", ""]
    if runner_name:
        lines += [f"Runner: {format_table_cell(runner_name)}", ""]
    lines += ["Rates use backend timings and natural EOS termination. "
              "Each row compares identical weights and requested MTP settings. "
              "Ratios are omitted when either case fails. MTP on requests Lemonade defaults; "
              "actual drafting is not verified.", ""]
    for scenario in dict.fromkeys(key[2] for key in h):
        lines += [f"## {format_table_cell(scenario)}", "",
                  "| Model / test mode | Context | Vulkan tok/s | HRX tok/s | HRX / Vulkan | Vulkan result | HRX result |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for key, (hm, hr) in h.items():
            if key[2] != scenario:
                continue
            vm, vr = v[key]
            has_hrx_rate = hr is not None
            has_vulkan_denominator = vr not in (None, 0)
            hrx_case_passed = hm["result"] == "pass"
            vulkan_case_passed = vm["result"] == "pass"
            comparable = (has_hrx_rate and has_vulkan_denominator
                          and hrx_case_passed and vulkan_case_passed)
            ratio = f"{hr / vr * 100:.1f}%" if comparable else "—"
            rate_text = lambda rate: "—" if rate is None else f"{rate:.1f}"
            outcome = lambda model: (f"{model['result'].upper()} (report only)"
                if model["outcome"] == "REPORT_ONLY" else model["outcome"])
            cells = [key[0], str(key[1]), rate_text(vr), rate_text(hr), ratio,
                     outcome(vm), outcome(hm)]
            lines.append("| " + " | ".join(format_table_cell(c) for c in cells) + " |")
        lines.append("")
    errors = [(backend, model["model"], error)
              for backend, doc in (("HRX", hrx), ("Vulkan", vulkan))
              for model in doc["models"] for error in model["errors"]]
    if errors:
        lines += ["", "## Recorded failures", ""]
        lines.extend(f"- {format_table_cell(backend + ' ' + name + ': ' + error)}"
                     for backend, name, error in errors)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hrx_benchmark", type=Path)
    parser.add_argument("vulkan_benchmark", type=Path)
    args = parser.parse_args()
    return write_report("Lemonade benchmark", lambda: format_report(
        load_json(args.hrx_benchmark), load_json(args.vulkan_benchmark),
        runner_name=os.environ.get("RUNNER_NAME")))


if __name__ == "__main__":
    raise SystemExit(main())
