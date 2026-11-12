#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Render the streaming long-context perplexity artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_report import UNAVAILABLE_MEASUREMENT, format_code, format_table_cell, load_json, write_report


def _ppl(measurement: dict[str, Any]) -> str:
    if measurement["status"] != "ok":
        return "unsupported" if measurement["status"] == "unsupported" else UNAVAILABLE_MEASUREMENT
    return f"{measurement['ppl']['value']:.4f}"


def format_report(document: dict[str, Any]) -> str:
    corpus = document["corpus"]
    settings = document["settings"]
    lines = [
        "## Long-context teacher-forced perplexity",
        "",
        f"**Corpus:** {format_code(corpus['name'])} (sha256 {format_code(corpus['sha256'][:12])}) · "
        f"**Maximum HRX/Vulkan ratio:** {settings['max_perplexity_ratio']:g}",
        f"**Sources:** automation {format_code(document['source_revisions']['automation'][:12])} · "
        f"llama.cpp {format_code(document['source_revisions']['llama_cpp'][:12])} · "
        f"HRX {format_code(document['source_revisions']['hrx_system'][:12])}",
        "",
        "Each context row prefills exactly N − scored tokens from the corpus, then scores the next "
        "tokens by teacher-forced continuation with full-vocabulary likelihood. The evaluator retains "
        "one vocabulary-logit row at a time. Contexts above a model's declared training context are "
        "reported as unsupported; allocation failures remain failures.",
        "",
        "| Model | Context | Prefix / scored | HRX PPL | Vulkan PPL | HRX/Vulkan |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    failures = []
    for model in document["models"]:
        for context, pair in model["contexts"].items():
            hrx = pair["hrx"]
            vulkan = pair["vulkan"]
            ratio = pair["ratio"]
            comparison = "—" if ratio is None else f"{ratio:.4f} ({pair['numerical_verdict'].upper()})"
            lines.append(
                f"| {format_code(model['model'])} | {int(context):,} | "
                f"{hrx.get('prefix_tokens', int(context) - document['contexts'][context]['scored_tokens']):,} / "
                f"{document['contexts'][context]['scored_tokens']:,} | {_ppl(hrx)} | {_ppl(vulkan)} | {comparison} |"
            )
            for backend, measurement in (("HRX", hrx), ("Vulkan", vulkan)):
                if measurement["status"] == "failed":
                    failures.append(
                        f"- {format_code(model['model'])} context {context} {backend}: "
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
    return write_report("long-context perplexity", lambda: format_report(load_json(args.artifact)))


if __name__ == "__main__":
    raise SystemExit(main())
