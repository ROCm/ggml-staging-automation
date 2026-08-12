# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "hrx"))

import write_lemonade_benchmark_report as report  # noqa: E402


def make_scenario(
    name: str,
    *,
    output_tokens: int = 16,
    tps: float = 20.0,
    ttft: float = 5.0,
    vram: float = 1.0,
) -> dict:
    return {
        "name": name,
        "output_tokens": output_tokens,
        "ttft_ms": {"mean": ttft, "min": ttft - 1, "max": ttft + 1},
        "tps": {"mean": tps, "min": tps - 1, "max": tps + 1},
        "vram_peak_gb": vram,
    }


def make_benchmark(*scenarios: dict) -> dict:
    return {
        "models": [
            {
                "model": "example-model",
                "results": [
                    {
                        "recipe": "llamacpp",
                        "backend": "server",
                        "ctx_size": 4096,
                        "backend_args": "",
                        "scenarios": list(scenarios),
                    }
                ],
            }
        ]
    }


class ComparisonFormattingTest(unittest.TestCase):
    def test_default_comparison_output_is_unchanged(self) -> None:
        left = make_benchmark(make_scenario("chat", tps=20, ttft=5, vram=1))
        right = make_benchmark(make_scenario("chat", tps=10, ttft=6, vram=2))

        actual = report.format_comparison_table(report.match_scenarios(left, right))

        expected = """## Lemonade HRX/Vulkan comparison

TPS parity is HRX mean TPS as a percentage of Vulkan mean TPS.

### `example-model`

**Recipe:** `llamacpp` · **Context:** `4096` tokens · **Backend arguments:** _(none)_

| Scenario | Output tokens | HRX TPS mean | Vulkan TPS mean | TPS parity | HRX TTFT mean (ms) | Vulkan TTFT mean (ms) | HRX VRAM peak (GB) | Vulkan VRAM peak (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat | 16 | 20.0 | 10.0 | 200.0% | 5.0 | 6.0 | 1.0 | 2.0 |"""
        self.assertEqual(actual, expected)

    def test_custom_title_and_backend_labels_cover_all_columns(self) -> None:
        left = make_benchmark(make_scenario("chat", tps=15, ttft=7, vram=3))
        right = make_benchmark(make_scenario("chat", tps=20, ttft=8, vram=4))
        matches = report.match_scenarios(
            left,
            right,
            left_backend_label="Current HRX",
            right_backend_label="Main HRX",
        )

        actual = report.format_comparison_table(
            matches,
            title="Custom comparison",
            left_backend_label="Current HRX",
            right_backend_label="Main HRX",
        )

        self.assertIn("## Custom comparison", actual)
        self.assertIn(
            "TPS parity is Current HRX mean TPS as a percentage of Main HRX mean TPS.",
            actual,
        )
        for metric in ("TPS mean", "TTFT mean (ms)", "VRAM peak (GB)"):
            self.assertIn(f"Current HRX {metric}", actual)
            self.assertIn(f"Main HRX {metric}", actual)
        self.assertIn("| chat | 16 | 15.0 | 20.0 | 75.0% | 7.0 | 8.0 | 3.0 | 4.0 |", actual)

    def test_main_comparisons_match_reordered_scenarios(self) -> None:
        current_hrx = make_benchmark(
            make_scenario("first", tps=12),
            make_scenario("second", tps=24),
        )
        main_hrx = make_benchmark(
            make_scenario("second", tps=12),
            make_scenario("first", tps=6),
        )
        current_vulkan = make_benchmark(
            make_scenario("first", tps=30),
            make_scenario("second", tps=40),
        )
        main_vulkan = make_benchmark(
            make_scenario("second", tps=20),
            make_scenario("first", tps=20),
        )

        actual = report.format_main_comparisons(
            current_hrx,
            current_vulkan,
            main_hrx,
            main_vulkan,
            "https://github.example/actions/runs/123",
        )

        self.assertIn(
            "[CI run on `main`](https://github.example/actions/runs/123)", actual
        )
        hrx_table, vulkan_table = actual.split(
            "## Lemonade current Vulkan/main Vulkan comparison"
        )
        self.assertLess(hrx_table.index("| first |"), hrx_table.index("| second |"))
        self.assertLess(
            vulkan_table.index("| first |"), vulkan_table.index("| second |")
        )
        self.assertIn("| first | 16 | 12.0 | 6.0 | 200.0%", hrx_table)
        self.assertIn("| second | 16 | 40.0 | 20.0 | 200.0%", vulkan_table)

    def test_same_artifact_smoke_has_identical_metrics_and_full_parity(self) -> None:
        benchmark = make_benchmark(
            make_scenario("first", tps=12, ttft=7, vram=3),
            make_scenario("second", tps=24, ttft=9, vram=4),
        )

        actual = report.format_main_comparisons(
            benchmark,
            benchmark,
            benchmark,
            benchmark,
        )

        self.assertEqual(actual.count("100.0%"), 4)
        self.assertEqual(
            actual.count("| first | 16 | 12.0 | 12.0 | 100.0% | 7.0 | 7.0 | 3.0 | 3.0 |"),
            2,
        )
        self.assertEqual(
            actual.count("| second | 16 | 24.0 | 24.0 | 100.0% | 9.0 | 9.0 | 4.0 | 4.0 |"),
            2,
        )


class MatchingIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = make_benchmark(make_scenario("chat"))

    def test_missing_scenario_on_either_side_is_rejected(self) -> None:
        empty = make_benchmark()
        for left, right, missing_label in (
            (self.benchmark, empty, "missing from Right"),
            (empty, self.benchmark, "missing from Left"),
        ):
            with self.subTest(missing_label=missing_label):
                with self.assertRaisesRegex(
                    report.BenchmarkReportError, missing_label
                ):
                    report.match_scenarios(
                        left,
                        right,
                        left_backend_label="Left",
                        right_backend_label="Right",
                    )

    def test_duplicate_scenario_on_either_side_is_rejected(self) -> None:
        duplicate = make_benchmark(make_scenario("chat"), make_scenario("chat"))
        for left, right, duplicate_label in (
            (duplicate, self.benchmark, "Duplicate Left comparison key"),
            (self.benchmark, duplicate, "Duplicate Right comparison key"),
        ):
            with self.subTest(duplicate_label=duplicate_label):
                with self.assertRaisesRegex(
                    report.BenchmarkReportError, duplicate_label
                ):
                    report.match_scenarios(
                        left,
                        right,
                        left_backend_label="Left",
                        right_backend_label="Right",
                    )

    def test_output_token_mismatch_is_rejected(self) -> None:
        mismatch = make_benchmark(make_scenario("chat", output_tokens=17))
        with self.assertRaisesRegex(
            report.BenchmarkReportError, "Left=16, Right=17"
        ):
            report.match_scenarios(
                self.benchmark,
                mismatch,
                left_backend_label="Left",
                right_backend_label="Right",
            )


class ReportCliTest(unittest.TestCase):
    def invoke(self, current_hrx: dict, current_vulkan: dict, *args: str):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            hrx_path = directory / "current-hrx.json"
            vulkan_path = directory / "current-vulkan.json"
            hrx_path.write_text(json.dumps(current_hrx), encoding="utf-8")
            vulkan_path.write_text(json.dumps(current_vulkan), encoding="utf-8")

            resolved_args = [
                str(directory / value.removeprefix("TMP/"))
                if value.startswith("TMP/")
                else value
                for value in args
            ]
            stdout = StringIO()
            stderr = StringIO()
            with mock.patch.object(
                sys,
                "argv",
                ["write_lemonade_benchmark_report.py", str(hrx_path), str(vulkan_path), *resolved_args],
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = report.main()
            return return_code, stdout.getvalue(), stderr.getvalue(), directory

    def test_two_file_usage_keeps_current_report_and_adds_fallback(self) -> None:
        benchmark = make_benchmark(make_scenario("chat"))
        return_code, stdout, stderr, _ = self.invoke(benchmark, benchmark)

        self.assertEqual(return_code, 0)
        self.assertIn("## Lemonade HRX/Vulkan comparison", stdout)
        self.assertIn(report.BASELINE_UNAVAILABLE_MESSAGE, stdout)
        self.assertEqual(stderr, "")

    def test_valid_baselines_add_both_tables_and_run_link(self) -> None:
        benchmark = make_benchmark(make_scenario("chat"))
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            current_hrx = directory / "current-hrx.json"
            current_vulkan = directory / "current-vulkan.json"
            main_hrx = directory / "main-hrx.json"
            main_vulkan = directory / "main-vulkan.json"
            for path in (current_hrx, current_vulkan, main_hrx, main_vulkan):
                path.write_text(json.dumps(benchmark), encoding="utf-8")

            stdout = StringIO()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "write_lemonade_benchmark_report.py",
                    str(current_hrx),
                    str(current_vulkan),
                    "--baseline-hrx-benchmark",
                    str(main_hrx),
                    "--baseline-vulkan-benchmark",
                    str(main_vulkan),
                    "--baseline-run-url",
                    "https://github.example/actions/runs/123",
                ],
            ), redirect_stdout(stdout):
                return_code = report.main()

        self.assertEqual(return_code, 0)
        self.assertIn("Lemonade current HRX/main HRX comparison", stdout.getvalue())
        self.assertIn(
            "Lemonade current Vulkan/main Vulkan comparison", stdout.getvalue()
        )
        self.assertIn("https://github.example/actions/runs/123", stdout.getvalue())
        self.assertNotIn(report.BASELINE_UNAVAILABLE_MESSAGE, stdout.getvalue())

    def test_each_historical_integrity_failure_falls_back_atomically(self) -> None:
        current = make_benchmark(make_scenario("chat"))
        missing = make_benchmark()
        duplicate = make_benchmark(make_scenario("chat"), make_scenario("chat"))
        token_mismatch = make_benchmark(make_scenario("chat", output_tokens=17))
        invalid_baselines = (
            (missing, current),
            (current, duplicate),
            (token_mismatch, current),
        )

        for main_hrx_benchmark, main_vulkan_benchmark in invalid_baselines:
            with self.subTest(
                main_hrx=main_hrx_benchmark,
                main_vulkan=main_vulkan_benchmark,
            ):
                with TemporaryDirectory() as temporary_directory:
                    directory = Path(temporary_directory)
                    main_hrx = directory / "main-hrx.json"
                    main_vulkan = directory / "main-vulkan.json"
                    main_hrx.write_text(
                        json.dumps(main_hrx_benchmark), encoding="utf-8"
                    )
                    main_vulkan.write_text(
                        json.dumps(main_vulkan_benchmark), encoding="utf-8"
                    )
                    stdout = StringIO()
                    stderr = StringIO()
                    current_hrx = directory / "current-hrx.json"
                    current_vulkan = directory / "current-vulkan.json"
                    current_hrx.write_text(json.dumps(current), encoding="utf-8")
                    current_vulkan.write_text(json.dumps(current), encoding="utf-8")
                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "write_lemonade_benchmark_report.py",
                            str(current_hrx),
                            str(current_vulkan),
                            "--baseline-hrx-benchmark",
                            str(main_hrx),
                            "--baseline-vulkan-benchmark",
                            str(main_vulkan),
                        ],
                    ), redirect_stdout(stdout), redirect_stderr(stderr):
                        return_code = report.main()

                self.assertEqual(return_code, 0)
                self.assertIn(report.BASELINE_UNAVAILABLE_MESSAGE, stdout.getvalue())
                self.assertNotIn("current HRX/main HRX", stdout.getvalue())
                self.assertNotIn("current Vulkan/main Vulkan", stdout.getvalue())
                self.assertIn("Could not compare with main benchmark", stderr.getvalue())

    def test_baseline_load_failure_falls_back_without_partial_tables(self) -> None:
        benchmark = make_benchmark(make_scenario("chat"))
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            current_hrx = directory / "current-hrx.json"
            current_vulkan = directory / "current-vulkan.json"
            valid_baseline = directory / "valid.json"
            malformed_baseline = directory / "malformed.json"
            missing_baseline = directory / "missing.json"
            for path in (current_hrx, current_vulkan, valid_baseline):
                path.write_text(json.dumps(benchmark), encoding="utf-8")
            malformed_baseline.write_text("{", encoding="utf-8")

            baseline_pairs = (
                (missing_baseline, valid_baseline),
                (malformed_baseline, valid_baseline),
                (valid_baseline, malformed_baseline),
            )
            for main_hrx, main_vulkan in baseline_pairs:
                with self.subTest(main_hrx=main_hrx, main_vulkan=main_vulkan):
                    stdout = StringIO()
                    stderr = StringIO()
                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "write_lemonade_benchmark_report.py",
                            str(current_hrx),
                            str(current_vulkan),
                            "--baseline-hrx-benchmark",
                            str(main_hrx),
                            "--baseline-vulkan-benchmark",
                            str(main_vulkan),
                        ],
                    ), redirect_stdout(stdout), redirect_stderr(stderr):
                        return_code = report.main()

                    self.assertEqual(return_code, 0)
                    self.assertIn("## Lemonade HRX/Vulkan comparison", stdout.getvalue())
                    self.assertIn(
                        report.BASELINE_UNAVAILABLE_MESSAGE, stdout.getvalue()
                    )
                    self.assertNotIn("current HRX/main HRX", stdout.getvalue())
                    self.assertNotIn("current Vulkan/main Vulkan", stdout.getvalue())
                    self.assertIn(
                        "Could not compare with main benchmark", stderr.getvalue()
                    )

    def test_partial_baseline_arguments_fall_back(self) -> None:
        benchmark = make_benchmark(make_scenario("chat"))

        return_code, stdout, stderr, _ = self.invoke(
            benchmark,
            benchmark,
            "--baseline-hrx-benchmark",
            "TMP/main-hrx.json",
        )

        self.assertEqual(return_code, 0)
        self.assertIn(report.BASELINE_UNAVAILABLE_MESSAGE, stdout)
        self.assertNotIn("current HRX/main HRX", stdout)
        self.assertIn("both baseline benchmark paths are required", stderr)

    def test_baseline_numeric_overflow_falls_back(self) -> None:
        current = make_benchmark(make_scenario("chat"))
        overflowing = make_benchmark(make_scenario("chat", tps=10**400))
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            paths = {
                "current_hrx": directory / "current-hrx.json",
                "current_vulkan": directory / "current-vulkan.json",
                "main_hrx": directory / "main-hrx.json",
                "main_vulkan": directory / "main-vulkan.json",
            }
            for name, path in paths.items():
                benchmark = overflowing if name == "main_hrx" else current
                path.write_text(json.dumps(benchmark), encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "write_lemonade_benchmark_report.py",
                    str(paths["current_hrx"]),
                    str(paths["current_vulkan"]),
                    "--baseline-hrx-benchmark",
                    str(paths["main_hrx"]),
                    "--baseline-vulkan-benchmark",
                    str(paths["main_vulkan"]),
                ],
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = report.main()

        self.assertEqual(return_code, 0)
        self.assertIn(report.BASELINE_UNAVAILABLE_MESSAGE, stdout.getvalue())
        self.assertNotIn("current HRX/main HRX", stdout.getvalue())
        self.assertIn("Could not compare with main benchmark", stderr.getvalue())

    def test_current_integrity_failure_remains_gating(self) -> None:
        current_hrx = make_benchmark(make_scenario("chat"))
        current_vulkan = deepcopy(current_hrx)
        current_vulkan["models"][0]["results"][0]["scenarios"][0][
            "output_tokens"
        ] = 17

        return_code, stdout, stderr, _ = self.invoke(current_hrx, current_vulkan)

        self.assertEqual(return_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Could not write Lemonade benchmark report", stderr)


if __name__ == "__main__":
    unittest.main()
