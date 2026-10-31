#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Render video benchmark correctness and measured server timings for CI."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from benchmark_report import format_table_cell, load_json, write_report


def measured_average(case: dict, key: str) -> str:
    values = []
    for sample in case["samples"]:
        is_measured = sample["phase"] == "measured"
        has_metric = key in sample
        if is_measured and has_metric:
            values.append(sample[key])
    return f"{mean(values):.1f}" if values else "—"


def case_status(case: dict | None) -> str:
    if case is None:
        return "FAIL"
    return "PASS" if case["passed"] else "FAIL"


def case_metric(case: dict | None, key: str) -> str:
    return measured_average(case, key) if case is not None else "—"


def format_report(document: dict) -> str:
    lines = [
        "## Video and image checks",
        "",
        (
            "Every warmup and all three measured answers must be correct. "
            "The table reports mean measured wall latency and llama-server generation "
            "tok/s; timing has no pass threshold."
        ),
        "",
    ]
    if not document["models"]:
        return "\n".join([*lines, "No video-enabled models were selected."])
    lines.extend(
        [
            "| Model | Case | Vulkan | HRX | Vulkan wall ms | HRX wall ms | Vulkan tok/s | HRX tok/s |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    failures = []
    for model in document["models"]:
        hrx_cases = {case["id"]: case for case in model["backends"]["hrx"]["cases"]}
        vulkan_cases = {
            case["id"]: case for case in model["backends"]["vulkan"]["cases"]
        }
        for case_id in dict.fromkeys([*vulkan_cases, *hrx_cases]):
            hrx = hrx_cases.get(case_id)
            vulkan = vulkan_cases.get(case_id)
            cells = [
                model["model"],
                case_id,
                case_status(vulkan),
                case_status(hrx),
                case_metric(vulkan, "wall_ms"),
                case_metric(hrx, "wall_ms"),
                case_metric(vulkan, "predicted_tps"),
                case_metric(hrx, "predicted_tps"),
            ]
            lines.append(
                "| " + " | ".join(format_table_cell(str(cell)) for cell in cells) + " |"
            )
            for backend, case in (("Vulkan", vulkan), ("HRX", hrx)):
                if case is None:
                    failures.append(
                        f"- {model['model']} {backend} {case_id}: no result."
                    )
                    continue
                for sample in case["samples"]:
                    if sample["correct"]:
                        continue
                    answer = sample.get("answer", sample.get("error", "no answer"))
                    failures.append(
                        f"- {model['model']} {backend} {case_id} "
                        f"{sample['phase']} {sample['index']}: expected {case['expected']}, "
                        f"got {answer!r}."
                    )
        for backend, result in model["backends"].items():
            if "error" in result:
                failures.append(f"- {model['model']} {backend}: {result['error']}.")
    if failures:
        lines.extend(["", "### Failures", "", *failures])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_results", type=Path)
    args = parser.parse_args()
    return write_report(
        "video benchmark", lambda: format_report(load_json(args.video_results))
    )


if __name__ == "__main__":
    raise SystemExit(main())
