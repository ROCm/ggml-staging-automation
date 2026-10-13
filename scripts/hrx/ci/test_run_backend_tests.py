#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for externally produced device listings and test logs."""

import unittest

from run_backend_tests import hrx_device, test_evidence


class BackendEvidenceTests(unittest.TestCase):
    def test_exact_visible_device(self):
        self.assertEqual(hrx_device("  HRX0: AMD Strix Halo (gfx1151) (98304 MiB)\n"), "HRX0")
        for listing in ("Vulkan0: GPU\n", "HRX0: AMD (gfx1201)\n", "HRX0: AMD (gfx1151)\nHRX1: AMD (gfx1151)\n"):
            with self.subTest(listing=listing), self.assertRaises(RuntimeError):
                hrx_device(listing)

    def test_all_skipped_is_not_execution(self):
        with self.assertRaises(RuntimeError):
            test_evidence("Backend 1/2: HRX0\n  Skipping\nBackend 2/2: CPU\n  Skipping\n2/2 backends passed\nOK\n", "HRX0")

    def test_upstream_summary_includes_unsupported_cases(self):
        evidence = test_evidence(
            "Backend 1/2: HRX0\n  Device description: AMD (gfx1151)\n"
            "  ADD(type=f32): \x1b[1;32mOK\x1b[0m\n"
            "  MUL(type=f32): not supported [HRX0] \n"
            "  2/2 tests passed\n  Backend HRX0: OK\n"
            "Backend 2/2: CPU\n  Skipping\n2/2 backends passed\nOK\n", "HRX0")
        self.assertEqual(evidence["executed"], 1)
        self.assertEqual(evidence["unsupported"], 1)
        self.assertTrue(evidence["suite_completed"])

    def test_failed_and_truncated_suite(self):
        log = "Backend 1/2: HRX0\n  Device description: AMD (gfx1151)\n  ADD(type=f32): FAIL\n"
        evidence = test_evidence(log, "HRX0")
        self.assertEqual(evidence["failed"], 1)
        self.assertEqual(evidence["failing_cases"], ["ADD(type=f32)"])
        self.assertFalse(evidence["suite_completed"])


if __name__ == "__main__":
    unittest.main()
