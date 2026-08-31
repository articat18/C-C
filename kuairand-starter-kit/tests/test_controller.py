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
        self.assertEqual(result["global_comparison"], result["comparison"])
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
                "bpr_hybrid",
                "Automatically allocate the next ID.",
                parameters={"bpr_weight": 0.5},
            )

        self.assertEqual(spec.experiment_id, "E0003")
        self.assertEqual(path, experiments / "E0003" / "spec.json")
        self.assertTrue(path.is_file())
        self.assertEqual(spec.schema_version, 2)
        self.assertEqual(spec.stage, "loss")
        self.assertEqual(spec.operator, "none")
        self.assertAlmostEqual(spec.parameters["bpr_weight"], 0.5)

    def test_status_uses_published_baseline_as_initial_best(self):
        controller = ExperimentController(
            ExperimentRegistry(self.root / "index.jsonl")
        )
        status = controller.status()
        self.assertEqual(status["best_source"], "stable_published_baseline")
        self.assertAlmostEqual(status["baseline_valid_primary"], 0.6016)
        self.assertAlmostEqual(status["best_valid_primary"], 0.6016)
        self.assertIsNone(status["best_candidate_decision"])

    def test_matched_comparison_requires_identical_control_settings(self):
        registry = ExperimentRegistry(self.root / "index.jsonl")
        controller = ExperimentController(registry)
        control_parameters = {
            "embedding_dim": 16,
            "learning_rate": 0.001,
            "l2": 1e-6,
            "patience": 4,
            "batch_size": 8192,
        }
        registry.append({
            "experiment_id": "E0030",
            "status": "success",
            "template": "pointwise_fm",
            "stage": "training",
            "operator": "none",
            "seed": 0,
            "parameters": control_parameters,
            "budget": {"max_epochs": 40, "max_wall_seconds": 21600},
            "data_dir": "./KuaiRand-Pure/data",
            "metrics": {"valid": {"primary": 0.6015}},
        })
        spec = ExperimentSpec.from_mapping({
            "schema_version": 2,
            "experiment_id": "E0031",
            "template": "pointwise_fm",
            "stage": "features",
            "operator": "date_period_bucket",
            "control_experiment_id": "E0030",
            "hypothesis": "Add date periods.",
            "evidence": "Temporal drift is material.",
            "expected_effect": "Improve ranking over a matched pointwise control.",
            "parameters": {},
        })
        comparison = controller._matched_comparison(
            spec, {"metrics": {"valid": {"primary": 0.6021}}}
        )
        self.assertEqual(comparison["reference"], "E0030")
        self.assertAlmostEqual(comparison["improvement"], 0.0006)
        self.assertEqual(comparison["decision"], "promising")

        mismatched = ExperimentSpec.from_mapping({
            **spec.to_dict(),
            "experiment_id": "E0032",
            "seed": 1,
        })
        with self.assertRaisesRegex(Exception, "seed"):
            controller._validated_control(mismatched)

    def test_lambdarank_can_use_same_pipeline_pointwise_control(self):
        registry = ExperimentRegistry(self.root / "lambda-index.jsonl")
        controller = ExperimentController(registry)
        parameters = {
            "embedding_dim": 16,
            "learning_rate": 0.001,
            "l2": 1e-6,
            "patience": 4,
            "batch_size": 8192,
        }
        registry.append({
            "experiment_id": "E0031",
            "status": "success",
            "template": "pointwise_fm",
            "stage": "features",
            "operator": "date_period_bucket",
            "seed": 0,
            "parameters": parameters,
            "budget": {"max_epochs": 40, "max_wall_seconds": 21600},
            "data_dir": "./KuaiRand-Pure/data",
            "metrics": {"valid": {"primary": 0.6021}},
        })
        spec = ExperimentSpec.from_mapping({
            "schema_version": 2,
            "experiment_id": "E0040",
            "template": "lambdarank_fm",
            "stage": "loss",
            "operator": "date_period_bucket",
            "control_experiment_id": "E0031",
            "hypothesis": "Fine-tune the matched pointwise model.",
            "evidence": "Top-five ordering has headroom.",
            "expected_effect": "Improve primary through weighted pairs.",
            "parameters": {},
        })
        self.assertEqual(
            controller._validated_control(spec)["experiment_id"], "E0031"
        )

    def test_sequence_model_can_use_matched_pointwise_control(self):
        registry = ExperimentRegistry(self.root / "sequence-index.jsonl")
        controller = ExperimentController(registry)
        parameters = {
            "embedding_dim": 16, "learning_rate": 0.001, "l2": 1e-6,
            "patience": 4, "batch_size": 8192,
        }
        registry.append({
            "experiment_id": "E0001", "status": "success", "template": "pointwise_fm",
            "stage": "training", "operator": "none", "seed": 0,
            "parameters": parameters, "budget": {"max_epochs": 40, "max_wall_seconds": 21600},
            "data_dir": "./KuaiRand-Pure/data", "metrics": {"valid": {"primary": 0.6016}},
        })
        spec = ExperimentSpec.from_mapping({
            "schema_version": 2, "experiment_id": "E0002", "template": "sequence_mlp",
            "stage": "model", "operator": "none", "control_experiment_id": "E0001",
            "hypothesis": "Use causal prior-positive-video history.",
            "evidence": "History features show a measurable signal.",
            "expected_effect": "Improve ranking with nonlinear history interactions.",
            "parameters": {},
        })
        self.assertEqual(controller._validated_control(spec)["experiment_id"], "E0001")


if __name__ == "__main__":
    unittest.main()
