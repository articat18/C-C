import unittest

from experiment_boundary import (
    BoundaryViolation,
    assert_model_selection_split,
    assert_protected_files_unchanged,
    resolve_editable_path,
)


class ExperimentBoundaryTests(unittest.TestCase):
    def test_protected_hashes_match(self):
        assert_protected_files_unchanged()

    def test_candidate_path_is_editable(self):
        path = resolve_editable_path("candidates/new_model.py")
        self.assertTrue(str(path).endswith("candidates/new_model.py"))

    def test_experiment_engine_path_is_editable(self):
        path = resolve_editable_path("experiment_engine/controller.py")
        self.assertTrue(str(path).endswith("experiment_engine/controller.py"))

    def test_evaluator_is_protected(self):
        with self.assertRaises(BoundaryViolation):
            resolve_editable_path("evaluate.py")

    def test_path_outside_repository_is_rejected(self):
        with self.assertRaises(BoundaryViolation):
            resolve_editable_path("../outside.py")

    def test_validation_is_the_only_model_selection_split(self):
        assert_model_selection_split("valid")
        with self.assertRaises(BoundaryViolation):
            assert_model_selection_split("test")


if __name__ == "__main__":
    unittest.main()
