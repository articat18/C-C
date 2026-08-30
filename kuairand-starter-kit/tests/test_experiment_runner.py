from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiment_engine.checkpoints import CheckpointManager
from experiment_engine.experiment_runner import run_experiment
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_boundary import REPOSITORY_ROOT


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="runner-test-", dir=runs)

    def tearDown(self):
        self.temporary.cleanup()

    def test_runner_seals_test_split_and_writes_checkpoint(self):
        spec = ExperimentSpec.from_mapping(
            {
                "schema_version": 2,
                "experiment_id": "E0099",
                "template": "bpr_hybrid",
                "stage": "cleaning",
                "operator": "missing_duration_category",
                "hypothesis": "Exercise the deterministic runner contract.",
                "evidence": "Zero duration needs a distinct missing category.",
                "expected_effect": "Separate missing duration from genuine short videos.",
                "parameters": {"popularity_weight": 0.0},
                "budget": {"max_epochs": 1, "max_wall_seconds": 60},
            }
        )
        model = SimpleNamespace(
            V=np.ones((4, 2), dtype=np.float32),
            W=np.zeros(4, dtype=np.float32),
            b=np.float32(0),
            predict=lambda features: np.asarray([0.9, 0.1], dtype=np.float32),
        )

        def fake_fit(splits, **kwargs):
            self.assertEqual(splits["test"], [])
            self.assertEqual(kwargs["epochs"], 1)
            encoded, _ = kwargs["encode_fn"](splits)
            self.assertEqual(encoded["train"][0].shape[1], 6)
            self.assertEqual(encoded["valid"][0].shape[1], 6)
            return model, encoded

        loaded = {
            "train": [(20220408, "u", "v1", "a", "t", 1.0, 1)],
            "valid": [
                (20220422, "u", "v1", "a", "t", 1.0, 1),
                (20220422, "u", "v2", "a", "t", 1.0, 0),
            ],
            "test": [(20220429, "sealed", "sealed", "a", "t", 1.0, 1)],
        }
        manager = CheckpointManager(Path(self.temporary.name) / "experiments")
        with mock.patch(
            "experiment_engine.experiment_runner.load", return_value=loaded
        ) as data_load, mock.patch(
            "experiment_engine.experiment_runner.baseline_models._fit_fm_bpr",
            side_effect=fake_fit,
        ):
            result = run_experiment(
                spec, checkpoint_manager=manager, verbose=False
            )

        data_load.assert_called_once_with(
            str(spec.resolved_data_dir()), split_names=("train", "valid")
        )
        self.assertEqual(result["selection_split"], "valid")
        self.assertEqual(result["metrics"]["valid"]["primary"], 1.0)
        self.assertNotIn("test", result["metrics"])
        self.assertEqual(len(result["checkpoints"]), 1)
        self.assertEqual(result["stage"], "cleaning")
        self.assertEqual(result["operator"], "missing_duration_category")
        self.assertEqual(result["encoded_fields"][-1], "duration_missing")
        self.assertIn("baseline_comparison", result["diagnostics"])
        self.assertIn("validation_subgroups", result["diagnostics"])
        self.assertTrue(result["diagnostics"]["validation_subgroups"])

    def test_runner_applies_train_only_duplicate_weights_without_dropping_rows(self):
        spec = ExperimentSpec.from_mapping(
            {
                "schema_version": 2,
                "experiment_id": "E0098",
                "template": "bpr_hybrid",
                "stage": "cleaning",
                "operator": "inverse_duplicate_frequency",
                "hypothesis": "Repeated interactions should contribute once in expectation.",
                "evidence": "The training split contains exact feature duplicates.",
                "expected_effect": "Reduce duplicate-driven sampling bias.",
                "parameters": {},
                "budget": {"max_epochs": 1, "max_wall_seconds": 60},
            }
        )
        model = SimpleNamespace(
            V=np.ones((4, 2), dtype=np.float32),
            W=np.zeros(4, dtype=np.float32),
            b=np.float32(0),
            predict=lambda features: np.asarray([0.9, 0.1], dtype=np.float32),
        )
        loaded = {
            "train": [
                (20220408, "u", "v1", "a", "t", 1.0, 1),
                (20220408, "u", "v1", "a", "t", 1.0, 1),
                (20220408, "u", "v2", "a", "t", 1.0, 0),
            ],
            "valid": [
                (20220422, "u", "v1", "a", "t", 1.0, 1),
                (20220422, "u", "v2", "a", "t", 1.0, 0),
            ],
        }

        def fake_fit(splits, **kwargs):
            encoded, _ = kwargs["encode_fn"](splits)
            np.testing.assert_array_equal(
                kwargs["train_sample_weights"], np.asarray([0.5, 0.5, 1.0])
            )
            self.assertEqual(len(encoded["train"][0]), 3)
            self.assertEqual(len(encoded["valid"][0]), 2)
            return model, encoded

        manager = CheckpointManager(Path(self.temporary.name) / "experiments")
        with mock.patch(
            "experiment_engine.experiment_runner.load", return_value=loaded
        ), mock.patch(
            "experiment_engine.experiment_runner.baseline_models._fit_fm_bpr",
            side_effect=fake_fit,
        ):
            result = run_experiment(spec, checkpoint_manager=manager, verbose=False)

        self.assertEqual(result["rows"]["train"], 3)
        self.assertEqual(result["encoded_fields"], [
            "user_id", "video_id", "author_id", "tab", "dur_bucket",
        ])
        diagnostics = result["operator_diagnostics"]
        self.assertEqual(diagnostics["duplicate_groups"], 1)
        self.assertEqual(diagnostics["effective_training_mass"], 2.0)


if __name__ == "__main__":
    unittest.main()
