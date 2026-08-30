import json
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
        package = self.root / "experiments" / spec.experiment_id
        package.mkdir(parents=True)
        (package / "spec.json").write_text(
            json.dumps(spec.to_dict()), encoding="utf-8"
        )

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

        self.assertEqual(result["comparison"]["decision"], "reject_or_refine")
        self.assertEqual(
            result["comparison"]["reference"], "stable_published_baseline"
        )
        self.assertAlmostEqual(result["comparison"]["previous_best"], 0.6016)
        self.assertTrue((package / "spec.json").is_file())
        self.assertTrue((package / "result.json").is_file())
        records = list(registry.records())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["experiment_id"], "E0088")
        self.assertEqual(records[0]["status"], "success")

    def test_create_skips_ids_already_present_in_specs(self):
        registry = ExperimentRegistry(self.root / "index.jsonl")
        controller = ExperimentController(registry)
        experiments = self.root / "experiments"
        templates = experiments / "templates"
        templates.mkdir(parents=True)
        (templates / "one.json").write_text(
            '{"experiment_id":"E0001"}\n', encoding="utf-8"
        )
        (templates / "two.json").write_text(
            '{"experiment_id":"E0002"}\n', encoding="utf-8"
        )

        def sandbox_path(value):
            path = Path(value)
            if path == Path("experiments"):
                return experiments
            return path

        with mock.patch(
            "experiment_engine.controller.resolve_editable_path",
            side_effect=sandbox_path,
        ):
            spec, path = controller.create(
                "bpr_hybrid", "Automatically allocate the next ID."
            )

        self.assertEqual(spec.experiment_id, "E0003")
        self.assertEqual(path, experiments / "E0003" / "spec.json")
        self.assertTrue(path.is_file())

    def test_status_uses_published_baseline_as_initial_best(self):
        controller = ExperimentController(
            ExperimentRegistry(self.root / "index.jsonl")
        )
        status = controller.status()
        self.assertEqual(status["best_source"], "stable_published_baseline")
        self.assertAlmostEqual(status["baseline_valid_primary"], 0.6016)
        self.assertAlmostEqual(status["best_valid_primary"], 0.6016)
        self.assertIsNone(status["best_candidate_decision"])


if __name__ == "__main__":
    unittest.main()
