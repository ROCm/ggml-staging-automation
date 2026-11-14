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
        f"**Requested contexts:** {', '.join(f'{n:,}' if isinstance(n, int) else n for n in settings['context_lengths'])} · "
        f"**Prefill tokens:** up to {settings['prefill_tokens']:,} · "
        f"**Decode tokens:** {settings['decode_tokens']} · "
        f"**Warmups:** {settings['warmups']} · **Measured repetitions:** {settings['repetitions']}",
        f"**Sources:** automation {format_code(document['source_revisions']['automation'][:12])} · "
        f"llama.cpp {format_code(document['source_revisions']['llama_cpp'][:12])} · "
        f"HRX {format_code(document['source_revisions']['hrx_system'][:12])}",
        "",
        "Prefill measures the final prompt tokens after llama-bench populates the earlier "
        "context outside the timer. Decode uses `tg T @ d (N − T)`: llama-bench "
        "populates the history before timing T generated tokens. Contexts above each model's GGUF-declared "
        "limit are skipped. Hardware allocation failures are failures. Only explicit "
        "context/backend `fail` expectations classify model failures "
        "as XFAIL; infrastructure failures are never excused.",
    ]
    if any(model["contexts"] for model in document["models"]):
        lines.extend([
            "",
            "| Model | Context | GGUF context | HRX pp tok/s | Vulkan pp tok/s | pp ratio | HRX tg tok/s | Vulkan tg tok/s | tg ratio | HRX outcome | Vulkan outcome |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ])
    failures = []
    for model in document["models"]:
        declared_context = model["model_context"]
        declared_context_text = f"{declared_context:,}" if declared_context is not None else "unknown"
        for context, backends in model["contexts"].items():
            hrx = backends["hrx"]
            vulkan = backends["vulkan"]
            context_text = f"{int(context):,}" if context != "max" else "max"
            lines.append(
                f"| {format_code(model['model'])} | {context_text} | "
                f"{declared_context_text} | "
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
    skipped = []
    for model in document["models"]:
        if model["model_context"] is None:
            continue
        lengths = [length for length in settings["context_lengths"]
                   if isinstance(length, int) and length > model["model_context"]]
        if lengths:
            skipped.append(
                f"- {format_code(model['model'])}: {', '.join(f'{length:,}' for length in lengths)} "
                f"(GGUF limit {model['model_context']:,})."
            )
    if skipped:
        lines.extend(["", "### Skipped above model context", "", *skipped])
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
