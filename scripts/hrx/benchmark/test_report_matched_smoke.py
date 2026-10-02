"""Exercise statistical gates and fixed coverage at the report input boundary."""

import copy
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import fmean, stdev

import report_matched_smoke as report


def fixture():
    models = [f"model-{number:02}" for number in range(25)]
    records = []
    perplexity = []
    for model in models:
        for scenario in report.SCENARIOS:
            for variant in report.VARIANTS:
                for backend in report.BACKENDS:
                    for number in report.ROUNDS:
                        mean = 103.0 if variant == "after" else 100.0
                        if backend == "vulkan":
                            mean = 200.0
                        records.append({"model": model, "scenario": scenario, "variant": variant,
                                        "backend": backend, "round": number, "status": "ok",
                                        "samples_tps": [mean - 1, mean, mean + 1], "mean_tps": mean})
        for regime in report.REGIMES:
            for variant in report.VARIANTS:
                for backend in report.BACKENDS:
                    perplexity.append({"model": model, "regime": regime, "variant": variant,
                                       "backend": backend, "status": "ok", "value": 10.25})
    return {"schema_version": 1, "expected_models": models, "scenarios": list(report.SCENARIOS),
            "records": records, "perplexity": perplexity, "status": "complete", "failures": []}


class MatchedSmokeReportTests(unittest.TestCase):
    def test_cli_exit_code_tracks_acceptance_and_preserves_failed_report(self):
        for complete in (True, False):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                data = fixture()
                if not complete:
                    del data["records"][0]
                root = Path(directory)
                raw = root / "matched-smoke.json"
                raw.write_text(json.dumps(data))
                result = subprocess.run([sys.executable, report.__file__, "--input", str(raw),
                                         "--output-dir", str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if complete else 2, result.stderr)
                self.assertEqual(json.loads(result.stdout)["all_pass"], complete)
                self.assertTrue((root / "report.md").is_file())
                self.assertEqual(len(json.loads((root / "summary.json").read_text())["throughput"]), 50)

    def test_complete_evidence_and_outputs(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            summary = report.write_report(data, directory)
            self.assertTrue(summary["all_pass"])
            self.assertTrue(summary["execution_complete"])
            self.assertTrue(summary["evidence_complete"])
            self.assertEqual(summary["classifications"], {"improved": 50})
            self.assertEqual(summary["coverage"]["nonincreasing_perplexity_comparisons"], 100)
            self.assertEqual(len(summary["throughput"]), 50)
            self.assertEqual(len(summary["perplexity"]), 100)
            self.assertAlmostEqual(summary["throughput"][0]["normalized_delta_percent"], 3.0)
            output = Path(directory)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            for name, count in (("throughput.csv", 50), ("perplexity.csv", 100)):
                with (output / name).open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
            self.assertIn("different estimator", (output / "report.md").read_text())

    def test_arithmetic_pooling_and_paired_interval_are_distinct(self):
        data = fixture()
        speeds = {("before", "hrx"): [50.0, 100.0, 200.0],
                  ("before", "vulkan"): [100.0, 100.0, 100.0],
                  ("after", "hrx"): [60.0, 110.0, 180.0],
                  ("after", "vulkan"): [110.0, 90.0, 120.0]}
        for observation in data["records"]:
            if observation["model"] != "model-00":
                continue
            mean = speeds[observation["variant"], observation["backend"]][observation["round"]]
            observation["mean_tps"] = mean
            observation["samples_tps"] = [mean - 1, mean, mean + 1]
        summary = report.build_report(data)
        row = summary["throughput"][0]
        self.assertAlmostEqual(row["normalized_delta_percent"], -6.25)
        self.assertAlmostEqual(row["raw_hrx_delta_percent"], 0)
        changes = [math.log((60 / 110) / (50 / 100)),
                   math.log((110 / 90) / (100 / 100)),
                   math.log((180 / 120) / (200 / 100))]
        center = fmean(changes)
        half_width = 4.3026527299 * stdev(changes) / math.sqrt(3)
        self.assertAlmostEqual(row["paired_geometric_delta_percent"], 100 * math.expm1(center))
        self.assertAlmostEqual(row["ci95_lower_percent"], 100 * math.expm1(center - half_width))
        self.assertAlmostEqual(row["ci95_upper_percent"], 100 * math.expm1(center + half_width))
        self.assertEqual(row["classification"], "inconclusive")
        self.assertFalse(summary["all_pass"])
        random.Random(47).shuffle(data["records"])
        random.Random(48).shuffle(data["perplexity"])
        shuffled = report.build_report(data)
        self.assertEqual(shuffled["throughput"], summary["throughput"])
        self.assertEqual(shuffled["perplexity"], summary["perplexity"])

    def test_classification_boundaries(self):
        cases = [(0.001, 1, "improved"), (0, 1, "within-margin"),
                 (-2, 1, "within-margin"), (-2.0001, -2, "inconclusive"),
                 (-3, -2.0001, "regressed"), (-3, 1, "inconclusive")]
        for lower, upper, expected in cases:
            with self.subTest(lower=lower, upper=upper):
                self.assertEqual(report.classify_interval(lower, upper), expected)

    def test_missing_observations_do_not_shrink_denominators(self):
        data = fixture()
        del data["records"][0]
        del data["perplexity"][0]
        summary = report.build_report(data)
        self.assertFalse(summary["all_pass"])
        self.assertFalse(summary["evidence_complete"])
        self.assertEqual(len(summary["throughput"]), 50)
        self.assertEqual(len(summary["perplexity"]), 100)
        self.assertEqual(summary["coverage"]["valid_throughput_comparisons"], 49)
        self.assertEqual(summary["coverage"]["valid_perplexity_comparisons"], 99)
        self.assertEqual(summary["throughput"][0]["classification"], "incomplete")

    def test_invalid_throughput_observations_are_rejected(self):
        changes = [{"samples_tps": [float("nan"), 100, 100]},
                   {"samples_tps": [float("inf"), 100, 100]},
                   {"samples_tps": [0, 100, 100]},
                   {"samples_tps": [True, 100, 100]},
                   {"samples_tps": [100, 100]},
                   {"mean_tps": 99}, {"mean_tps": float("nan")},
                   {"status": "error", "error": "worker failed"},
                   {"round": True}, {"backend": "unknown"}]
        for change in changes:
            with self.subTest(change=change):
                data = fixture()
                data["records"][0].update(change)
                summary = report.build_report(data)
                self.assertFalse(summary["all_pass"])
                self.assertEqual(summary["coverage"]["valid_throughput_comparisons"], 49)
                json.dumps(summary, allow_nan=False)

    def test_invalid_perplexity_and_increases_block_acceptance(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            with self.subTest(value=value):
                data = fixture()
                data["perplexity"][0]["value"] = value
                summary = report.build_report(data)
                self.assertFalse(summary["all_pass"])
                self.assertEqual(summary["coverage"]["valid_perplexity_comparisons"], 99)
        data = fixture()
        observation = next(item for item in data["perplexity"] if item["variant"] == "after")
        observation["value"] += 0.0001
        summary = report.build_report(data)
        self.assertTrue(summary["evidence_complete"])
        self.assertFalse(summary["all_pass"])
        self.assertEqual(summary["coverage"]["nonincreasing_perplexity_comparisons"], 99)

    def test_duplicate_extra_and_incomplete_coverage_rejected(self):
        for kind in ("duplicate", "extra", "model-count", "status", "failures"):
            with self.subTest(kind=kind):
                data = fixture()
                if kind == "duplicate":
                    data["records"].append(copy.deepcopy(data["records"][0]))
                if kind == "extra":
                    extra = copy.deepcopy(data["records"][0])
                    extra["model"] = "unexpected-model"
                    data["records"].append(extra)
                if kind == "model-count":
                    data["expected_models"].pop()
                if kind == "status":
                    data["status"] = "incomplete"
                if kind == "failures":
                    data["failures"] = ["command failed"]
                summary = report.build_report(data)
                self.assertFalse(summary["all_pass"])
                self.assertFalse(summary["evidence_complete"])
                self.assertEqual(summary["coverage"]["expected_throughput_comparisons"], 50)
                self.assertEqual(summary["coverage"]["expected_perplexity_comparisons"], 100)


if __name__ == "__main__":
    unittest.main()
