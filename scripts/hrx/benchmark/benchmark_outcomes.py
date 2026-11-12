#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared outcome handling for benchmark measurements."""

from __future__ import annotations

from collections.abc import Iterable

MODEL_FAILURE_KINDS = frozenset({
    "model-context", "model-execution", "resource-exhausted", "numerical",
})


def aggregate_status(statuses: Iterable[str]) -> str:
    """Return fail unless every requested measurement completed successfully."""
    values = tuple(statuses)
    if not values or any(value in {"failed", "fail", "unsupported"} for value in values):
        return "fail"
    return "pass" if all(value in {"ok", "pass"} for value in values) else "fail"


def classify_outcome(
    result: str,
    expectation: str,
    *,
    expected_failure_allowed: bool = True,
    infrastructure_failure: bool = False,
) -> str:
    """Apply the manifest's pass/fail/report-only expectation to one mode."""
    if infrastructure_failure:
        return "FAIL"
    if expectation == "report-only":
        return "REPORT_ONLY"
    if expectation == "fail":
        if result == "pass":
            return "XPASS"
        return "XFAIL" if expected_failure_allowed else "FAIL"
    return "FAIL" if result == "fail" else "PASS"


def is_unexpected(outcome: str) -> bool:
    return outcome in ("FAIL", "XPASS")
