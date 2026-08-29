import unittest
from unittest import mock

import numpy as np

from experiment_engine.reproduce_baseline import compare_scores, reproduce


class BaselineReproductionTests(unittest.TestCase):
    def test_score_comparison_uses_absolute_tolerance(self):
        expected = {
            "valid": {"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.595},
            "test": {"GAUC": 0.65, "nDCG@5": 0.52, "primary": 0.585},
        }
        observed = {
            split: {metric: score + 0.001 for metric, score in metrics.items()}
            for split, metrics in expected.items()
        }
        self.assertTrue(compare_scores(observed, expected, 0.002)["passed"])
        observed["test"]["primary"] += 0.002
        self.assertFalse(compare_scores(observed, expected, 0.002)["passed"])

    def test_reproduction_report_normalizes_numpy_metrics(self):
        observed = {
            split: {
                "GAUC": np.float32(0.6610),
                "nDCG@5": np.float32(0.5282),
                "primary": np.float32(0.5946),
            }
            for split in ("valid", "test")
        }
        with mock.patch(
            "experiment_engine.reproduce_baseline.load", return_value={}
        ), mock.patch(
            "experiment_engine.reproduce_baseline.run_bpr_ensemble",
            return_value=observed,
        ), mock.patch(
            "experiment_engine.reproduce_baseline.compare_scores",
            return_value={"passed": True, "tolerance": 0.002, "checks": []},
        ):
            report = reproduce("./KuaiRand-Pure/data", verbose=False)
        self.assertIsInstance(report["observed"]["valid"]["GAUC"], float)


if __name__ == "__main__":
    unittest.main()
