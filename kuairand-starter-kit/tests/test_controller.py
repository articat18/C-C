from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiment_engine.controller import ExperimentController, _has_converged
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.registry import ExperimentRegistry
from experiment_boundary import REPOSITORY_ROOT


class ControllerConvergenceTests(unittest.TestCase):
    def test_requires_three_consecutive_weak_iterations(self):
        self.assertFalse(_has_converged([0.6000, 0.6010, 0.6015]))
        self.assertTrue(_has_converged([0.6000, 0.6010, 0.6015, 0.6018]))

    def test_material_improvement_resets_patience(self):
        self.assertFalse(_has_converged([0.6000, 0.6010, 0.6031, 0.6035]))


class ExperimentControllerTests(unittest.TestCase):
    def setUp(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="controller-test-", dir=runs)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_success_is_written_to_result_and_append_only_registry(self):
        registry = ExperimentRegistry(self.root / "index.jsonl")
        controller = ExperimentController(registry)
        spec = ExperimentSpec.from_mapping(
            {
                "schema_version": 1,
                "experiment_id": "E0088",
                "template": "bpr_hybrid",
                "hypothesis": "Verify controller persistence.",
                "budget": {"max_epochs": 1, "max_wall_seconds": 60},
            }
        )
        fake_result = {
            "experiment_id": "E0088",
            "status": "success",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "duration_seconds": 1.25,
            "metrics": {
                "valid": {"GAUC": 0.66, "nDCG@5": 0.54, "primary": 0.60}
            },
        }

        def sandbox_path(value):
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(self.root):
                return path
            relative = path.relative_to("experiments") if path != Path("experiments") else Path()
            return (self.root / "experiments" / relative).resolve()

        with mock.patch(
            "experiment_engine.controller.resolve_editable_path", side_effect=sandbox_path
        ), mock.patch(
            "experiment_engine.controller.run_experiment", return_value=fake_result
        ):
            result = controller.run(spec, verbose=False)

        self.assertEqual(result["comparison"]["decision"], "keep")
        self.assertTrue((self.root / "experiments" / "E0088" / "spec.json").is_file())
        self.assertTrue((self.root / "experiments" / "E0088" / "result.json").is_file())
        records = list(registry.records())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["experiment_id"], "E0088")
        self.assertEqual(records[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
