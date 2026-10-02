#!/usr/bin/env python3
"""Report paired HRX/Vulkan smoke evidence without hiding incomplete coverage.

The input is the matched-smoke driver's schema-1 artifact: three paired rounds
for each of 25 models, two workloads, two packages, and two backends, plus both
perplexity regimes. A normalized change compares HRX/Vulkan after versus before.
Arithmetic pooled throughput describes the observed speeds; a separate paired
log-ratio estimator and Student-t interval determine the 2% nonregression gate.

``build_report`` validates the artifact once and retains all 50 throughput and
100 perplexity comparisons, including missing or failed ones. ``write_report``
writes JSON, Markdown, and CSV evidence. The CLI returns nonzero unless every
throughput interval meets the margin and every perplexity value is nonincreasing.
Three rounds provide limited precision; inconclusive evidence never counts as a
pass, and these per-comparison intervals are not a simultaneous confidence claim.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, stdev


SCENARIOS = ("chat-short", "chat-long-output")
VARIANTS = ("before", "after")
BACKENDS = ("hrx", "vulkan")
REGIMES = ("prefill-like", "decode-like")
ROUNDS = (0, 1, 2)
MODEL_COUNT = 25
MARGIN_PERCENT = -2.0
T_95_DF2 = 4.3026527299


def positive_finite(value):
    is_number = isinstance(value, (int, float))
    is_bool = isinstance(value, bool)
    numeric = is_number and not is_bool
    if not numeric:
        return False
    finite = math.isfinite(value)
    positive = value > 0
    return finite and positive


def classify_interval(lower, upper):
    if lower > 0:
        return "improved"
    if lower >= MARGIN_PERCENT:
        return "within-margin"
    if upper < MARGIN_PERCENT:
        return "regressed"
    return "inconclusive"


def paired_statistics(before_ratios, after_ratios):
    changes = [math.log(after) - math.log(before)
               for before, after in zip(before_ratios, after_ratios)]
    center = fmean(changes)
    half_width = T_95_DF2 * stdev(changes) / math.sqrt(len(ROUNDS))
    lower = 100 * math.expm1(center - half_width)
    upper = 100 * math.expm1(center + half_width)
    return {
        "paired_round_delta_percent": [100 * math.expm1(x) for x in changes],
        "paired_geometric_delta_percent": 100 * math.expm1(center),
        "ci95_lower_percent": lower,
        "ci95_upper_percent": upper,
        "classification": classify_interval(lower, upper),
    }


def load_index(data, field, models, issues):
    """Reject invalid raw observations at the artifact provenance boundary."""
    throughput = field == "records"
    dimensions = ("model", "scenario", "variant", "backend", "round") if throughput else (
        "model", "regime", "variant", "backend")
    allowed = {
        "model": models,
        "scenario": SCENARIOS,
        "variant": VARIANTS,
        "backend": BACKENDS,
        "round": ROUNDS,
        "regime": REGIMES,
    }
    observations = data.get(field)
    if not isinstance(observations, list):
        issues.append(f"{field}: expected an array")
        return {}
    index = {}
    for number, observation in enumerate(observations):
        label = f"{field}[{number}]"
        if not isinstance(observation, dict):
            issues.append(f"{label}: expected an object")
            continue
        bad_dimensions = []
        for dimension in dimensions:
            value = observation.get(dimension)
            valid_type = isinstance(value, str)
            if dimension == "round":
                valid_type = type(value) is int
            if not valid_type:
                bad_dimensions.append(dimension)
                continue
            if value not in allowed[dimension]:
                bad_dimensions.append(dimension)
        if bad_dimensions:
            issues.append(f"{label}: invalid dimensions {', '.join(bad_dimensions)}")
            continue
        key = tuple(observation[name] for name in dimensions)
        if key in index:
            issues.append(f"{label}: duplicate observation {key}")
            index[key] = None
            continue
        errors = []
        if observation.get("status") != "ok":
            errors.append(f"status {observation.get('status')!r}: {observation.get('error', '')}")
        if throughput:
            samples = observation.get("samples_tps")
            samples_is_list = isinstance(samples, list)
            samples_valid = False
            if samples_is_list:
                count_valid = len(samples) == 3
                values_valid = all(positive_finite(value) for value in samples)
                samples_valid = count_valid and values_valid
            if not samples_valid:
                errors.append("expected exactly three positive finite throughput samples")
            mean = observation.get("mean_tps")
            mean_valid = positive_finite(mean)
            if not mean_valid:
                errors.append("mean_tps must be positive and finite")
            can_compare_mean = samples_valid and mean_valid
            if can_compare_mean:
                if not math.isclose(mean, fmean(samples), rel_tol=1e-9):
                    errors.append("mean_tps differs from the arithmetic sample mean")
        elif not positive_finite(observation.get("value")):
            errors.append("perplexity must be positive and finite")
        index[key] = None if errors else observation
        issues.extend(f"{label} {key}: {error}" for error in errors)
    return index


def throughput_comparison(model, scenario, index, issues):
    row = {"model": model, "scenario": scenario, "valid": False,
           "classification": "incomplete"}
    keys = [(model, scenario, variant, backend, round_number)
            for variant in VARIANTS for backend in BACKENDS for round_number in ROUNDS]
    missing = [key for key in keys if index.get(key) is None]
    if missing:
        row["missing_or_invalid_observations"] = [list(key) for key in missing]
        issues.append(f"throughput {model}/{scenario}: {len(missing)} missing or invalid observations")
        return row
    means = {}
    ratios = {}
    for variant in VARIANTS:
        for backend in BACKENDS:
            samples = [sample for round_number in ROUNDS
                       for sample in index[model, scenario, variant, backend, round_number]["samples_tps"]]
            means[variant, backend] = fmean(samples)
            row[f"{variant}_{backend}_tps"] = means[variant, backend]
        ratios[variant] = means[variant, "hrx"] / means[variant, "vulkan"]
        row[f"{variant}_hrx_vulkan_ratio"] = ratios[variant]
        row[f"{variant}_gap_percent"] = 100 * (ratios[variant] - 1)
    round_ratios = {
        variant: [index[model, scenario, variant, "hrx", number]["mean_tps"] /
                  index[model, scenario, variant, "vulkan", number]["mean_tps"]
                  for number in ROUNDS]
        for variant in VARIANTS
    }
    row.update({
        "normalized_delta_percent": 100 * (ratios["after"] / ratios["before"] - 1),
        "gap_change_percentage_points": 100 * (ratios["after"] - ratios["before"]),
        "raw_hrx_delta_percent": 100 * (means["after", "hrx"] / means["before", "hrx"] - 1),
        "raw_vulkan_delta_percent": 100 * (means["after", "vulkan"] / means["before", "vulkan"] - 1),
        "round_hrx_vulkan_ratios": round_ratios,
        "valid": True,
    })
    row.update(paired_statistics(round_ratios["before"], round_ratios["after"]))
    return row


def perplexity_comparison(model, regime, backend, index, issues):
    row = {"model": model, "regime": regime, "backend": backend,
           "valid": False, "nonincreasing": False}
    keys = [(model, regime, variant, backend) for variant in VARIANTS]
    missing = [key for key in keys if index.get(key) is None]
    if missing:
        row["missing_or_invalid_observations"] = [list(key) for key in missing]
        issues.append(f"perplexity {model}/{regime}/{backend}: {len(missing)} missing or invalid observations")
        return row
    before, after = [index[key]["value"] for key in keys]
    row.update({"valid": True, "before": before, "after": after,
                "delta_absolute": after - before,
                "delta_percent": 100 * (after / before - 1),
                "nonincreasing": after <= before})
    return row


def build_report(data):
    issues = []
    if not isinstance(data, dict):
        raise ValueError("matched-smoke artifact must be a JSON object")
    if data.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if data.get("status") != "complete":
        issues.append(f"driver status is {data.get('status')!r}")
    models = data.get("expected_models")
    if not isinstance(models, list):
        models = []
        issues.append("expected_models must be a list")
    valid_models = [model for model in models if isinstance(model, str) and model]
    models_unique = len(set(valid_models)) == len(models)
    count_valid = len(models) == MODEL_COUNT
    models_valid = models_unique and count_valid
    if not models_valid:
        issues.append("expected_models must contain exactly 25 unique nonempty model IDs")
    models = list(dict.fromkeys(valid_models))
    if data.get("scenarios") != list(SCENARIOS):
        issues.append(f"scenarios must be {list(SCENARIOS)!r}")
    failures = data.get("failures", [])
    if failures:
        issues.append(f"driver reports {len(failures)} failures")
    throughput_index = load_index(data, "records", models, issues)
    perplexity_index = load_index(data, "perplexity", models, issues)
    throughput = [throughput_comparison(model, scenario, throughput_index, issues)
                  for model in models for scenario in SCENARIOS]
    perplexity = [perplexity_comparison(model, regime, backend, perplexity_index, issues)
                  for model in models for regime in REGIMES for backend in BACKENDS]
    classifications = Counter(row["classification"] for row in throughput)
    throughput_passed = sum(row["classification"] in ("improved", "within-margin") for row in throughput)
    perplexity_passed = sum(row["nonincreasing"] for row in perplexity)
    evidence_complete = not issues
    all_throughput_passed = throughput_passed == 50
    all_perplexity_passed = perplexity_passed == 100
    all_pass = evidence_complete and all_throughput_passed and all_perplexity_passed
    return {
        "schema_version": 1,
        "all_pass": all_pass,
        "execution_complete": data.get("status") == "complete",
        "evidence_complete": evidence_complete,
        "method": {
            "arithmetic_delta": "100 * ((mean(HRX_after)/mean(Vulkan_after)) / (mean(HRX_before)/mean(Vulkan_before)) - 1)",
            "interval": "95% Student-t interval on three paired log-ratio changes, exponentiated to percent",
            "interval_estimator": "geometric mean of three paired after/before HRX/Vulkan ratios",
            "rounds": 3, "samples_per_invocation": 3, "degrees_of_freedom": 2,
            "t_critical": T_95_DF2, "nonregression_margin_percent": MARGIN_PERCENT,
            "simultaneous_confidence": False,
        },
        "coverage": {"expected_models": 25, "expected_throughput_comparisons": 50,
                     "valid_throughput_comparisons": sum(row["valid"] for row in throughput),
                     "passed_throughput_comparisons": throughput_passed,
                     "expected_perplexity_comparisons": 100,
                     "valid_perplexity_comparisons": sum(row["valid"] for row in perplexity),
                     "nonincreasing_perplexity_comparisons": perplexity_passed},
        "classifications": dict(classifications), "issues": issues,
        "driver_failures": failures, "metadata": data.get("metadata", {}),
        "throughput": throughput, "perplexity": perplexity,
    }


def number(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def markdown_report(summary):
    coverage = summary["coverage"]
    verdict = "PASS" if summary["all_pass"] else "NOT PASS"
    lines = [f"# Matched smoke comparison: {verdict}", "",
             "Throughput uses arithmetic means of nine samples per package/backend/workload. "
             "HRX/Vulkan gaps are `100 × (HRX/Vulkan − 1)`; normalized Δ is "
             "`100 × ((HRX_after/Vulkan_after)/(HRX_before/Vulkan_before) − 1)`.", "",
             "The paired estimate is the geometric mean of three round ratios. Its 95% interval uses "
             "Student-t on paired log-ratio changes (df=2, t=4.3026527299). It is a different estimator "
             "from the arithmetic pooled normalized Δ. Intervals are per comparison, not simultaneous.", "",
             "Improved: lower bound > 0%; within-margin: lower bound ≥ −2%; regressed: upper bound < −2%; "
             "otherwise inconclusive. Missing or failed evidence is incomplete. All 50 throughput comparisons "
             "must be improved/within-margin and all 100 perplexity comparisons must be nonincreasing to pass.", "",
             f"Coverage: {coverage['valid_throughput_comparisons']}/50 valid throughput comparisons; "
             f"{coverage['passed_throughput_comparisons']}/50 meet the throughput gate. "
             f"{coverage['valid_perplexity_comparisons']}/100 valid perplexity comparisons; "
             f"{coverage['nonincreasing_perplexity_comparisons']}/100 are nonincreasing.", "",
             "## Throughput", "",
             "| Model | Scenario | Before HRX | Before Vulkan | After HRX | After Vulkan | Gap before % | Gap after % | Normalized Δ % | Raw HRX Δ % | Round 1/2/3 Δ % | Paired estimate % | 95% interval % | Result |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for row in summary["throughput"]:
        fields = [row["model"], row["scenario"]]
        fields.extend(number(row.get(key)) for key in (
            "before_hrx_tps", "before_vulkan_tps", "after_hrx_tps", "after_vulkan_tps"))
        fields.extend(number(row.get(key), signed=True) for key in (
            "before_gap_percent", "after_gap_percent", "normalized_delta_percent", "raw_hrx_delta_percent"))
        rounds = row.get("paired_round_delta_percent")
        fields.append(" / ".join(number(value, signed=True) for value in rounds) if rounds else "—")
        fields.append(number(row.get("paired_geometric_delta_percent"), signed=True))
        interval = "—"
        if row["valid"]:
            interval = f"[{number(row['ci95_lower_percent'], True)}, {number(row['ci95_upper_percent'], True)}]"
        fields.extend((interval, row["classification"]))
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend(["", "## Perplexity", "",
                  "| Model | Regime | Backend | Before | After | Δ | Δ % | Result |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for row in summary["perplexity"]:
        result = "nonincreasing" if row["nonincreasing"] else "increased"
        if not row["valid"]:
            result = "incomplete"
        values = [f"{row[key]:.6f}" if key in row else "—" for key in ("before", "after", "delta_absolute")]
        fields = [row["model"], row["regime"], row["backend"], *values,
                  number(row.get("delta_percent"), True), result]
        lines.append("| " + " | ".join(fields) + " |")
    if summary["issues"]:
        lines.extend(["", "## Evidence issues", ""])
        lines.extend(f"- {issue}" for issue in summary["issues"])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_report(data, output_dir):
    summary = build_report(data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(markdown_report(summary))
    write_csv(output_dir / "throughput.csv", summary["throughput"])
    write_csv(output_dir / "perplexity.csv", summary["perplexity"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = write_report(json.loads(args.input.read_text()), args.output_dir)
    print(json.dumps({"all_pass": summary["all_pass"], "coverage": summary["coverage"],
                      "classifications": summary["classifications"]}, sort_keys=True))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
