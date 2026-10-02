#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check paired ratios and reject invalid external benchmark measurements."""

import json
import unittest

from run_paired_prefill import parse_rate, summarize


class PairedPrefillTests(unittest.TestCase):
    def test_shared_machine_slowdown_cancels_in_ratio(self):
        pairs = [{"hrx": 2 * rate, "vulkan": rate} for rate in [100, 50, 200]]
        result = summarize(pairs)
        self.assertAlmostEqual(result["hrx_over_vulkan"], 2)
        self.assertAlmostEqual(result["ratio_ci95_low"], 2)
        self.assertAlmostEqual(result["ratio_ci95_high"], 2)

    def test_interval_includes_observed_variation(self):
        result = summarize([{"hrx": rate, "vulkan": 100} for rate in [90, 100, 110]])
        self.assertLess(result["ratio_ci95_low"], 1)
        self.assertGreater(result["ratio_ci95_high"], 1)

    def test_parse_prefill(self):
        value = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4}]
        self.assertEqual(parse_rate(json.dumps(value)), 123.4)

    def test_reject_unusable_rates(self):
        for rate in [0, -1, float("nan"), float("inf")]:
            with self.subTest(rate=rate):
                value = [{"n_prompt": 512, "n_gen": 0, "avg_ts": rate}]
                with self.assertRaises(ValueError):
                    parse_rate(json.dumps(value))

    def test_reject_wrong_scenario(self):
        value = [{"n_prompt": 0, "n_gen": 512, "avg_ts": 123.4}]
        with self.assertRaises(ValueError):
            parse_rate(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
