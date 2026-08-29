import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import baseline
from data import encode
from experiment_engine.approval import (
    APPROVAL_PHRASE,
    ApprovalError,
    grant_final_approval,
)
from experiment_engine.checkpoints import CheckpointManager
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.finalize import finalize_experiment
from experiment_engine.registry import ExperimentRegistry
from experiment_boundary import REPOSITORY_ROOT


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="finalize-test-", dir=runs)
        self.root = Path(self.temporary.name)
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()
        self.registry = ExperimentRegistry(self.root / "index.jsonl")

    def tearDown(self):
        self.temporary.cleanup()

    def _sandbox_path(self, value):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(self.root):
            return path
        if path.parts and path.parts[0] in {"experiments", "runs"}:
            return (self.root / path).resolve()
        return path

    def _registered_spec(self):
        spec = ExperimentSpec.from_mapping(
            {
                "schema_version": 1,
                "experiment_id": "E0077",
                "template": "bpr_hybrid",
                "hypothesis": "Verify human-gated finalization.",
                "budget": {"max_epochs": 1, "max_wall_seconds": 60},
            }
        )
        directory = self.experiments / spec.experiment_id
        directory.mkdir()
        (directory / "spec.json").write_text(
            json.dumps(spec.to_dict()), encoding="utf-8"
        )
        self.registry.append(
            {
                "experiment_id": spec.experiment_id,
                "status": "success",
                "spec_fingerprint": spec.fingerprint(),
                "metrics": {"valid": {"primary": 0.6}},
            }
        )
        return spec

    def test_wrong_confirmation_is_rejected(self):
        with self.assertRaises(ApprovalError):
            grant_final_approval(
                "E0077",
                approved_by="teammate",
                confirmation="yes",
                registry=self.registry,
            )

    def test_finalization_requires_approval_before_loading_data(self):
        self._registered_spec()
        with mock.patch(
            "experiment_engine.finalize.resolve_editable_path",
            side_effect=self._sandbox_path,
        ), mock.patch(
            "experiment_engine.approval.resolve_editable_path",
            side_effect=self._sandbox_path,
        ), mock.patch("experiment_engine.finalize.load") as data_load:
            with self.assertRaises(ApprovalError):
                finalize_experiment("E0077", registry=self.registry)
        data_load.assert_not_called()

    def test_approved_experiment_can_generate_test_submission(self):
        spec = self._registered_spec()
        splits = {
            "train": [
                (20220408, "u", "v1", "a", "t", 1000.0, 1),
                (20220408, "u", "v2", "a", "t", 2000.0, 0),
            ],
            "valid": [
                (20220422, "u", "v1", "a", "t", 1000.0, 1),
                (20220422, "u", "v2", "a", "t", 2000.0, 0),
            ],
            "test": [
                (20220429, "u", "v1", "a", "t", 1000.0, 1),
                (20220429, "u", "v2", "a", "t", 2000.0, 0),
            ],
        }
        _, dimension = encode(splits)
        model = baseline.FM(dimension, k=16, seed=0)
        checkpoints = CheckpointManager(self.root / "checkpoints")
        checkpoints.save_member(
            spec.experiment_id,
            0,
            model=model,
            metadata={"spec_fingerprint": spec.fingerprint()},
        )

        with mock.patch(
            "experiment_engine.approval.resolve_editable_path",
            side_effect=self._sandbox_path,
        ):
            grant_final_approval(
                spec.experiment_id,
                approved_by="teammate",
                confirmation=APPROVAL_PHRASE,
                registry=self.registry,
            )

        submission = self.root / "runs" / "E0077" / "submission.csv"
        with mock.patch(
            "experiment_engine.finalize.resolve_editable_path",
            side_effect=self._sandbox_path,
        ), mock.patch(
            "experiment_engine.approval.resolve_editable_path",
            side_effect=self._sandbox_path,
        ), mock.patch(
            "experiment_engine.finalize.load", return_value=splits
        ):
            result = finalize_experiment(
                spec.experiment_id,
                submission_path=submission,
                registry=self.registry,
                checkpoint_manager=checkpoints,
            )

        self.assertEqual(result["status"], "finalized")
        self.assertIn("test", result["metrics"])
        self.assertTrue(submission.is_file())
        self.assertTrue(
            (self.experiments / spec.experiment_id / "final-result.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
