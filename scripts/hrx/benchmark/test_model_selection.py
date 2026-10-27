#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guard explicit selection against accidental expansion of ordinary CI coverage."""

import json
from pathlib import Path
import unittest

from run_batched_benchmark import BatchBenchmarkError, load_manifest, parse_args, select_models


MANIFEST = Path(__file__).resolve().parents[3] / "benchmarks/hrx/model_manifest.json"


class ModelSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)
        cls.flint_ids = {"flint-q3-k-m", "flint-iq2-xxs", "flint-iq2-xs", "flint-iq2-m", "flint-q2-k"}

    def test_ordinary_coverage_is_unchanged(self):
        for tier, count in (("single", 1), ("small", 25), ("all", 30)):
            with self.subTest(tier=tier):
                selected = select_models(self.manifest, tier)
                self.assertEqual(len(selected), count)
                self.assertFalse(self.flint_ids.intersection(model.id for model in selected))

    def test_explicit_ids_replace_tier_and_keep_manifest_order(self):
        requested = ["flint-iq2-m", "flint-q3-k-m"]
        for tier in ("single", "small", "all"):
            with self.subTest(tier=tier):
                selected = select_models(self.manifest, tier, json.dumps(requested))
                self.assertEqual([model.id for model in selected], list(reversed(requested)))

    def test_all_five_are_explicitly_selectable(self):
        selected = select_models(self.manifest, "single", json.dumps(sorted(self.flint_ids)))
        self.assertEqual({model.id for model in selected}, self.flint_ids)
        self.assertTrue(all(model.tier == "select-only" for model in selected))

    def test_invalid_selection_fails(self):
        for ids in ('null', '{}', '1', '[1]', '["missing"]', '["flint-q2-k", "flint-q2-k"]', 'bad'):
            with self.subTest(ids=ids), self.assertRaises(BatchBenchmarkError):
                select_models(self.manifest, "all", ids)
        for tier in ("select-only", "missing"):
            with self.subTest(tier=tier), self.assertRaises(BatchBenchmarkError):
                select_models(self.manifest, tier)

    def test_cli_defaults_and_explicit_selection(self):
        args = parse_args(["--benchmark-spec", "spec.json", "--model-manifest", str(MANIFEST),
                           "--work-root", "work", "--model-ids", '["flint-q2-k"]'])
        self.assertEqual(args.max_disk_gib, 60)
        self.assertEqual([model.id for model in select_models(self.manifest, args.model_tier, args.model_ids)],
                         ["flint-q2-k"])


if __name__ == "__main__":
    unittest.main()
