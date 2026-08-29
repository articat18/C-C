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
