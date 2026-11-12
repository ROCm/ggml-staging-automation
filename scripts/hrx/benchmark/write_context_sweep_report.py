#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Render independent prefill and populated-KV decode throughput results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_report import UNAVAILABLE_MEASUREMENT, format_code, format_table_cell, load_json, write_report


def _throughput(measurement: dict[str, Any]) -> str:
    if measurement["status"] == "unsupported":
        return "unsupported"
    if measurement["status"] != "ok":
        return UNAVAILABLE_MEASUREMENT
    return f"{measurement['tokens_per_second']:.2f}"


def _ratio(hrx: dict[str, Any], vulkan: dict[str, Any]) -> str:
    if hrx["status"] != "ok" or vulkan["status"] != "ok":
        return UNAVAILABLE_MEASUREMENT
    return f"{hrx['tokens_per_second'] / vulkan['tokens_per_second']:.3f}×"


def format_report(document: dict[str, Any]) -> str:
    settings = document["settings"]
    lines = [
        "## HRX/Vulkan llama-bench context sweep",
        "",
        f"**Contexts:** {', '.join(f'{n:,}' for n in settings['context_lengths'])} · "
        f"**Decode tokens:** {settings['decode_tokens']} · "
        f"**Warmups:** {settings['warmups']} · **Measured repetitions:** {settings['repetitions']}",
        f"**Sources:** automation {format_code(document['source_revisions']['automation'][:12])} · "
        f"llama.cpp {format_code(document['source_revisions']['llama_cpp'][:12])} · "
        f"HRX {format_code(document['source_revisions']['hrx_system'][:12])}",
        "",
        "Prefill uses a separate `pp N` run. Decode uses `tg T @ d (N − T)`: llama-bench "
        "prefills the history before starting the timer, then measures T generated tokens. The "
        "combined `-pg` result is not used. Contexts above the GGUF-declared training context "
        "retain an unsupported diagnostic and count as failed checks; hardware allocation failures "
        "are failures. Only explicit context/backend `fail` expectations classify model failures "
        "as XFAIL; infrastructure failures are never excused.",
        "",
        "| Model | Context | HRX pp tok/s | Vulkan pp tok/s | pp ratio | HRX tg tok/s | Vulkan tg tok/s | tg ratio | HRX outcome | Vulkan outcome |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    failures = []
    for model in document["models"]:
        for context, backends in model["contexts"].items():
            hrx = backends["hrx"]
            vulkan = backends["vulkan"]
            lines.append(
                f"| {format_code(model['model'])} | {int(context):,} | "
                f"{_throughput(hrx['prefill'])} | {_throughput(vulkan['prefill'])} | "
                f"{_ratio(hrx['prefill'], vulkan['prefill'])} | "
                f"{_throughput(hrx['decode'])} | {_throughput(vulkan['decode'])} | "
                f"{_ratio(hrx['decode'], vulkan['decode'])} | {hrx['outcome']} | {vulkan['outcome']} |"
            )
            for backend_name, measurements in (("HRX", hrx), ("Vulkan", vulkan)):
                for kind in ("prefill", "decode"):
                    measurement = measurements[kind]
                    if measurement["status"] == "failed":
                        failures.append(
                            f"- {format_code(model['model'])} context {context} {backend_name} {kind}: "
                            f"{measurement['failure_kind']} — {format_table_cell(measurement['error'])} "
                            f"(see {format_code(measurement['log'])})."
                        )
    if failures:
        lines.extend(["", "### Measurement failures", "", *failures])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    return write_report("context sweep", lambda: format_report(load_json(args.artifact)))


if __name__ == "__main__":
    raise SystemExit(main())
