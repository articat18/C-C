import unittest
from unittest import mock

import numpy as np

import baseline
from candidates.feature_pipeline import (
    FeatureOperatorError,
    encode_candidate_splits,
    encoded_field_names,
    operator_diagnostics,
    training_sample_weights,
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

    def test_inverse_duplicate_weights_preserve_alignment_and_ignore_labels(self):
        duplicate_train = [
            (20220408, "u1", "v1", "a1", "home", 1000.0, 1),
            (20220408, "u1", "v1", "a1", "home", 1000.0, 0),
            (20220408, "u1", "v2", "a1", "home", 1000.0, 0),
        ]
        splits = {"train": duplicate_train, "valid": self.valid, "test": []}
        weights = training_sample_weights(
            splits, operator_name="inverse_duplicate_frequency"
        )
        changed = dict(splits)
        changed["train"] = [row[:6] + (1 - row[6],) for row in duplicate_train]
        changed["valid"] = [row[:6] + (1 - row[6],) for row in self.valid]
        changed_weights = training_sample_weights(
            changed, operator_name="inverse_duplicate_frequency"
        )
        encoded, _ = encode_candidate_splits(
            splits, operator_name="inverse_duplicate_frequency"
        )

        np.testing.assert_array_equal(weights, np.asarray([0.5, 0.5, 1.0]))
        np.testing.assert_array_equal(weights, changed_weights)
        self.assertEqual(len(encoded["train"][0]), len(duplicate_train))
        self.assertEqual(len(encoded["valid"][0]), len(self.valid))
        self.assertEqual(encoded["train"][0].shape[1], 5)
        diagnostics = operator_diagnostics(
            splits, operator_name="inverse_duplicate_frequency"
        )
        self.assertEqual(diagnostics["duplicate_groups"], 1)
        self.assertEqual(diagnostics["excess_duplicate_rows"], 1)
        self.assertEqual(diagnostics["effective_training_mass"], 2.0)
        self.assertTrue(diagnostics["preserves_row_count"])

    def test_inverse_duplicate_operator_is_cleaning_only(self):
        validate_pipeline_selection("cleaning", "inverse_duplicate_frequency")
        with self.assertRaises(FeatureOperatorError):
            validate_pipeline_selection("features", "inverse_duplicate_frequency")

    def test_weighted_sampler_normalizes_pool_probabilities(self):
        rng = mock.Mock()
        rng.choice.return_value = np.asarray([2])
        chosen = baseline._sample_pool(
            rng,
            [0, 1, 2],
            1,
            np.asarray([0.5, 0.5, 1.0]),
        )

        np.testing.assert_array_equal(chosen, np.asarray([2]))
        _, kwargs = rng.choice.call_args
        np.testing.assert_allclose(kwargs["p"], np.asarray([0.25, 0.25, 0.5]))
        self.assertFalse(kwargs["replace"])

    def test_weighted_sampler_replaces_when_request_exceeds_pool(self):
        rng = mock.Mock()
        rng.choice.return_value = np.asarray([0, 0])
        baseline._sample_pool(
            rng,
            [0],
            2,
            np.asarray([0.5]),
        )

        _, kwargs = rng.choice.call_args
        self.assertTrue(kwargs["replace"])
        np.testing.assert_array_equal(kwargs["p"], np.asarray([1.0]))


if __name__ == "__main__":
    unittest.main()
