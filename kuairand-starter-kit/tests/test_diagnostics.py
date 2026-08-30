import unittest

import numpy as np

from experiment_engine.diagnostics import (
    baseline_comparison,
    profile_dataset,
    suggest_experiment_families,
    subgroup_metrics,
)
from experiment_boundary import REPOSITORY_ROOT
from data import load


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            (20220408, "u1", "v1", "a1", "home", 0.0, 1),
            (20220408, "u1", "v2", "a2", "home", 6000.0, 0),
            (20220409, "u2", "v1", "a1", "follow", 16000.0, 0),
        ]
        self.valid = [
            (20220422, "u1", "v1", "a1", "home", 0.0, 1),
            (20220423, "u3", "v3", "a3", "home", 2000.0, 0),
        ]

    def test_profile_is_deterministic_and_reports_drift(self):
        first = profile_dataset({"valid": self.valid, "train": self.train})
        second = profile_dataset({"train": self.train, "valid": self.valid})
        self.assertEqual(first, second)
        self.assertEqual(first["splits"]["train"]["rows"], 3)
        self.assertEqual(first["splits"]["train"]["duration"]["zero_rows"], 1)
        self.assertEqual(first["splits"]["valid"]["duplicate_user_video_rows"], 0)
        self.assertGreater(first["train_valid_drift"]["valid_unseen_user_rate"], 0.0)
        self.assertTrue(first["suggested_experiment_families"])

    def test_suggestions_have_stable_family_and_reason(self):
        profile = profile_dataset({"train": self.train, "valid": self.valid})
        suggestions = suggest_experiment_families(profile)
        self.assertTrue(all(set(item) == {"family", "reason"} for item in suggestions))

    def test_subgroups_and_baseline_deltas_are_structured(self):
        report = subgroup_metrics(self.valid, np.asarray([0.8, 0.1]), self.train)
        self.assertIn("user_activity/01-05", report)
        self.assertIn("duration/missing_zero", report)
        comparison = baseline_comparison(
            {"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65},
            {"GAUC": 0.66, "nDCG@5": 0.55, "primary": 0.605},
        )
        self.assertAlmostEqual(comparison["metrics"]["primary"]["delta"], 0.045)

    def test_subgroup_metrics_rejects_mismatched_predictions(self):
        with self.assertRaises(ValueError):
            subgroup_metrics(self.valid, np.asarray([0.8]), self.train)

    def test_empty_profile_has_safe_zero_statistics(self):
        profile = profile_dataset({"train": [], "valid": []})
        self.assertEqual(profile["splits"]["train"]["rows"], 0)
        self.assertEqual(profile["splits"]["train"]["positive_rate"], 0.0)
        self.assertEqual(profile["splits"]["valid"]["duration"]["max"], 0.0)

    def test_real_dataset_split_contract(self):
        data_dir = REPOSITORY_ROOT / "KuaiRand-Pure" / "data"
        required = data_dir / "log_standard_4_08_to_4_21_pure.csv"
        if not required.is_file():
            self.skipTest("KuaiRand-Pure data is not present")
        profile = profile_dataset(
            load(data_dir, split_names=("train", "valid"))
        )
        self.assertEqual(profile["splits"]["train"]["rows"], 1_141_112)
        self.assertEqual(profile["splits"]["valid"]["rows"], 124_909)
        self.assertEqual(
            profile["splits"]["train"]["duplicate_user_video_rows"], 48_362
        )
        self.assertEqual(profile["splits"]["valid"]["duplicate_user_video_rows"], 3_572)
        self.assertAlmostEqual(
            profile["train_valid_drift"]["valid_positive_rate_delta"],
            -0.0233358228,
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
