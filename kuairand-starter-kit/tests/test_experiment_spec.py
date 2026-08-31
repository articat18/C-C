import unittest

from experiment_engine.experiment_spec import ExperimentSpec, SpecificationError
from experiment_engine.experiment_templates import TemplateValidationError


def valid_spec():
    return {
        "schema_version": 1,
        "experiment_id": "E0042",
        "template": "bpr_hybrid",
        "hypothesis": "Pairwise training improves within-user ranking.",
        "parameters": {"bpr_weight": 0.5},
        "budget": {"max_epochs": 5, "max_wall_seconds": 60},
    }


class ExperimentSpecTests(unittest.TestCase):
    def test_applies_template_defaults_and_is_stable(self):
        spec = ExperimentSpec.from_mapping(valid_spec())
        self.assertEqual(spec.parameters["bpr_weight"], 0.5)
        self.assertEqual(spec.parameters["embedding_dim"], 16)
        self.assertEqual(spec.budget.max_epochs, 5)
        self.assertEqual(spec.fingerprint(), spec.fingerprint())
        self.assertNotIn("stage", spec.to_dict())

    def test_schema_two_accepts_one_reviewed_feature_operator(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "video_popularity_bucket",
            "evidence": "Rare and popular items have different subgroup performance.",
            "expected_effect": "Training-only popularity improves cold-item ordering.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.stage, "features")
        self.assertEqual(spec.operator, "video_popularity_bucket")
        self.assertEqual(spec.to_dict()["schema_version"], 2)

    def test_schema_two_records_matched_control(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "video_popularity_bucket",
            "evidence": "Rare and popular items have different subgroup performance.",
            "expected_effect": "Training-only popularity improves cold-item ordering.",
            "parameters": {},
            "control_experiment_id": "E0030",
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.control_experiment_id, "E0030")
        self.assertEqual(spec.to_dict()["control_experiment_id"], "E0030")

    def test_rejects_self_as_matched_control(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "video_popularity_bucket",
            "evidence": "Popularity differs.",
            "expected_effect": "Improve ranking.",
            "parameters": {},
            "control_experiment_id": "E0042",
        })
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_pointwise_template_uses_official_defaults(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "template": "pointwise_fm",
            "stage": "training",
            "operator": "none",
            "evidence": "Establish a governed matched control.",
            "expected_effect": "Reproduce the protected pointwise baseline.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.parameters["batch_size"], 8192)
        self.assertEqual(spec.parameters["l2"], 1e-6)

    def test_lambdarank_inherits_matched_feature_pipeline(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "template": "lambdarank_fm",
            "stage": "loss",
            "operator": "date_period_bucket",
            "control_experiment_id": "E0031",
            "evidence": "Date period passed paired replication.",
            "expected_effect": "Improve top-five ordering from the matched checkpoint.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.operator, "date_period_bucket")
        self.assertEqual(spec.control_experiment_id, "E0031")

    def test_lambdarank_requires_matched_control(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "template": "lambdarank_fm",
            "stage": "loss",
            "operator": "none",
            "evidence": "Ranking headroom remains.",
            "expected_effect": "Improve nDCG at five.",
            "parameters": {},
        })
        with self.assertRaisesRegex(SpecificationError, "matched pointwise control"):
            ExperimentSpec.from_mapping(value)

    def test_pointwise_ensemble_has_three_members(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "template": "pointwise_ensemble",
            "stage": "features",
            "operator": "date_period_bucket",
            "evidence": "Date period passed paired replication.",
            "expected_effect": "Reduce seed variance for the confirmed feature.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.template, "pointwise_ensemble")

    def test_schema_two_accepts_inverse_duplicate_frequency_operator(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "cleaning",
            "operator": "inverse_duplicate_frequency",
            "evidence": "Training contains repeated interactions.",
            "expected_effect": "Reduce repeated-row sampling bias.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.stage, "cleaning")
        self.assertEqual(spec.operator, "inverse_duplicate_frequency")

    def test_schema_two_accepts_smoothed_video_rate_operator(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "smoothed_video_long_view_rate",
            "evidence": "Item engagement rates differ in the training split.",
            "expected_effect": "Improve ordering with a smoothed item prior.",
            "parameters": {},
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.stage, "features")
        self.assertEqual(spec.operator, "smoothed_video_long_view_rate")

    def test_schema_two_rejects_operator_plus_scalar_change(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "video_popularity_bucket",
            "evidence": "Popularity subgroups differ.",
            "expected_effect": "Improve item ranking.",
            "parameters": {"bpr_weight": 0.5},
        })
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_schema_two_rejects_mismatched_stage(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "cleaning",
            "operator": "video_popularity_bucket",
            "evidence": "Popularity subgroups differ.",
            "expected_effect": "Improve item ranking.",
            "parameters": {},
        })
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_schema_two_rejects_two_scalar_changes(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "training",
            "operator": "none",
            "evidence": "Test one-change enforcement.",
            "expected_effect": "Only one scalar may differ.",
            "parameters": {"learning_rate": 0.002, "l2": 0.001},
        })
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_schema_two_records_validated_proposal_provenance(self):
        value = valid_spec()
        value.update({
            "schema_version": 2,
            "stage": "features",
            "operator": "video_popularity_bucket",
            "evidence": "Popularity subgroups differ.",
            "expected_effect": "Improve cold-item ordering.",
            "parameters": {},
            "provenance": {
                "proposal_fingerprint": "a" * 64,
                "context_fingerprint": "b" * 64,
                "source_model": "gemini-test",
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "manual_interventions": 1,
            },
        })
        spec = ExperimentSpec.from_mapping(value)
        self.assertEqual(spec.provenance["manual_interventions"], 1)
        self.assertEqual(spec.to_dict()["provenance"]["token_usage"]["input_tokens"], 10)

    def test_rejects_arbitrary_code_fields(self):
        value = valid_spec()
        value["command"] = "python arbitrary.py"
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_rejects_unknown_template(self):
        value = valid_spec()
        value["template"] = "module:function"
        with self.assertRaises(TemplateValidationError):
            ExperimentSpec.from_mapping(value)

    def test_rejects_unknown_template_parameter(self):
        value = valid_spec()
        value["parameters"] = {"python_source": "print('unsafe')"}
        with self.assertRaises(TemplateValidationError):
            ExperimentSpec.from_mapping(value)

    def test_rejects_data_outside_repository(self):
        value = valid_spec()
        value["data_dir"] = "/tmp/external-data"
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)

    def test_rejects_budget_above_controller_limit(self):
        value = valid_spec()
        value["budget"]["max_wall_seconds"] = 21601
        with self.assertRaises(SpecificationError):
            ExperimentSpec.from_mapping(value)


if __name__ == "__main__":
    unittest.main()
