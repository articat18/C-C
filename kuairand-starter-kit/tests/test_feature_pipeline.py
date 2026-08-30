import unittest

import numpy as np

from candidates.feature_pipeline import (
    FeatureOperatorError,
    encode_candidate_splits,
    encoded_field_names,
    validate_pipeline_selection,
)
from data import encode


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            (20220408, "u1", "v1", "a1", "home", 0.0, 1),
            (20220408, "u1", "v1", "a1", "home", 1000.0, 0),
            (20220408, "u2", "v2", "a2", "other", 2000.0, 1),
        ]
        self.valid = [
            (20220422, "u9", "v1", "a1", "home", 0.0, 0),
            (20220422, "u9", "unseen", "a9", "home", 9000.0, 1),
        ]

    def test_none_exactly_matches_protected_encoding(self):
        splits = {"train": self.train, "valid": self.valid, "test": []}
        expected, expected_dimension = encode(splits)
        actual, actual_dimension = encode_candidate_splits(splits)

        self.assertEqual(actual_dimension, expected_dimension)
        for split_name in splits:
            np.testing.assert_array_equal(actual[split_name][0], expected[split_name][0])
            np.testing.assert_array_equal(actual[split_name][1], expected[split_name][1])
            self.assertEqual(actual[split_name][2], expected[split_name][2])

    def test_operator_adds_one_train_fitted_field_and_ignores_valid_labels(self):
        splits = {"train": self.train, "valid": self.valid, "test": []}
        first, base_plus_operator = encode_candidate_splits(
            splits, operator_name="video_popularity_bucket"
        )
        changed = dict(splits)
        changed["valid"] = [row[:6] + (1 - row[6],) for row in self.valid]
        second, second_dimension = encode_candidate_splits(
            changed, operator_name="video_popularity_bucket"
        )

        self.assertEqual(first["train"][0].shape[1], 6)
        self.assertEqual(first["valid"][0].shape[1], 6)
        self.assertEqual(base_plus_operator, second_dimension)
        np.testing.assert_array_equal(first["valid"][0], second["valid"][0])
        self.assertEqual(encoded_field_names("video_popularity_bucket")[-1], "video_popularity")

    def test_stage_and_operator_must_match(self):
        validate_pipeline_selection("features", "video_popularity_bucket")
        with self.assertRaises(FeatureOperatorError):
            validate_pipeline_selection("cleaning", "video_popularity_bucket")


if __name__ == "__main__":
    unittest.main()
