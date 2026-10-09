# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Extract explicit backend failure evidence without guessing its cause.

Allocation errors require an allocation operation or an explicit out-of-memory
message. A generic compute error does not establish memory exhaustion. These
diagnostics describe log evidence; they never change benchmark expectations.
"""


def failure_kind(line: str) -> str | None:
    text = line.lower()
    out_of_memory = any(marker in text for marker in (
        "out of memory", "out_of_device_memory", "out_of_host_memory",
        "hiperroroutofmemory",
    ))
    allocation_operation = "allocat" in text
    failed_operation = any(marker in text for marker in (
        "failed", "failure", "resource_exhausted", "out_of_resources",
    ))
    allocation_failed = allocation_operation and failed_operation
    is_allocation_error = out_of_memory or allocation_failed
    if is_allocation_error:
        return "allocation"
    compute_operation = "graph_compute:" in text
    compute_failed = "failed" in text
    is_compute_error = compute_operation and compute_failed
    if is_compute_error:
        return "compute"
    return None


def model_diagnostics(log: str, models: list[str]) -> dict[str, list[dict[str, str]]]:
    """Keep the first example of each error kind during each model's lifetime.

    Lemonade loads models sequentially, including reloads after failed requests.
    Attribute evidence using its explicit load marker, not model-name substrings.
    Evidence is model-wide and includes warmup; it is not a per-scenario count.
    """
    evidence: dict[str, dict[str, str]] = {name: {} for name in models}
    current = ""
    for line in log.splitlines():
        _, marker, name = line.partition("(LlamaCpp) Loading model: ")
        if marker:
            current = name.strip()
        kind = failure_kind(line)
        known_model = current in evidence
        attributed_error = known_model and kind is not None
        if attributed_error:
            evidence[current].setdefault(kind, line.strip())
        if "(Router) Evicted model:" in line:
            current = ""
    return {
        name: [{"kind": kind, "message": message} for kind, message in errors.items()]
        for name, errors in evidence.items()
    }
