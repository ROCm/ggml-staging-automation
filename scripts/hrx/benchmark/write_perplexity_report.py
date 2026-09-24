#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Render one combined perplexity artifact as one current-run comparison table.

Each row shows prefill-like and decode-like HRX/Vulkan estimates, their ratios
and numerical verdicts, and the aggregate check. XFAIL and SKIP never hide raw
numerical failures. A failed Vulkan reference makes the check FAIL regardless
of the HRX expectation. Failed measurements remain unavailable in the table;
details below it identify the backend, regime, error kind, log, and model batch.

The CLI accepts one JSON path and writes the complete Markdown report to stdout.
Reading or formatting errors return status 1 without printing a partial report.
Recorded measurement failures remain reportable and do not fail the report CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_report import (
    UNAVAILABLE_MEASUREMENT,
    format_code,
    format_table_cell,
    load_json,
    write_report,
)


def format_ppl(measurement: dict[str, Any]) -> str:
    if measurement["status"] != "ok":
        return UNAVAILABLE_MEASUREMENT
    ppl = measurement["ppl"]
    return f"{ppl['value']:.4f} ± {ppl['uncertainty']:.4f}"


def format_comparison(pair: dict[str, Any]) -> str:
    """Keep numerical failures visible independently of model expectations."""
    if pair["numerical_verdict"] == "unavailable":
        return UNAVAILABLE_MEASUREMENT
    return f"{pair['ratio']:.4f} ({pair['numerical_verdict'].upper()})"


def format_check(row: dict[str, Any]) -> str:
    if row["reference_result"] == "fail":
        return f"FAIL (Vulkan); HRX {row['outcome']}"
    return row["outcome"]


def format_report(document: dict[str, Any]) -> str:
    """Render complete model rows and their failure evidence from one artifact."""
    corpus = document["corpus"]
    settings = document["settings"]
    extra_args = settings["extra_args"]
    arguments = (
        " ".join(format_code(arg) for arg in extra_args)
        if extra_args else "_(none)_"
    )
    lines = [
        "## HRX/Vulkan perplexity",
        "",
        f"**Corpus:** {format_code(corpus['name'])} "
        f"(sha256 {format_code(corpus['sha256'][:12])}) · "
        f"**Maximum HRX/Vulkan ratio:** {settings['max_perplexity_ratio']:g} · "
        f"**Extra llama-perplexity arguments:** {arguments}",
        "",
    ]
    for regime in document["regimes"].values():
        lines.append(
            f"- **{regime['name']}:** context {regime['ctx']}, "
            f"batch {regime['batch']}, microbatch {regime['microbatch']}, "
            f"chunks {regime['chunks']}."
        )
    lines.extend([
        "",
        "Ratios are HRX PPL divided by Vulkan PPL; equality at the limit passes. "
        "Numerical verdicts require valid measurements from both backends. "
        "`—` marks an unavailable measurement or comparison. The check applies "
        "the HRX expectation once across both regimes; Vulkan failures are always "
        "fatal. XFAIL and SKIP leave the measured numerical verdicts visible.",
        "",
    ])
    if not document["models"]:
        return "\n".join([*lines, "No models were benchmarked."])

    lines.extend([
        "| Model | Prefill HRX PPL | Prefill Vulkan PPL | Prefill ratio (verdict) | "
        "Decode HRX PPL | Decode Vulkan PPL | Decode ratio (verdict) | Check |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ])
    failures = []
    for row in document["models"]:
        cells = [format_code(row["model"])]
        for regime_id in ("prefill-like", "decode-like"):
            pair = row["regimes"][regime_id]
            cells.extend([
                format_ppl(pair["hrx"]),
                format_ppl(pair["vulkan"]),
                format_comparison(pair),
            ])
            for backend in ("hrx", "vulkan"):
                measurement = pair[backend]
                if measurement["status"] == "ok":
                    continue
                location = format_code(measurement["log"])
                if row["batch"] is not None:
                    location += f", batch {row['batch']}"
                failures.append(
                    f"- {format_code(row['model'])}, {regime_id}, {backend.upper()}: "
                    f"{measurement['failure_kind']} — "
                    f"{format_table_cell(measurement['error'])} (see {location})."
                )
        cells.append(format_check(row))
        lines.append("| " + " | ".join(cells) + " |")
    if failures:
        lines.extend(["", "### Measurement failures", "", *failures])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perplexity", type=Path)
    args = parser.parse_args()

    return write_report(
        "perplexity",
        lambda: format_report(load_json(args.perplexity)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
