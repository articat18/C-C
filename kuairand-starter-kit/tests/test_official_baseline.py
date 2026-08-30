import unittest

import numpy as np

from official_baseline import OfficialFM


class OfficialFMLogitsRegressionTests(unittest.TestCase):
    def test_logits_match_direct_formula_without_mutating_model(self):
        model = OfficialFM(dim=7, k=3, seed=11)
        model.W[:] = np.linspace(-0.2, 0.3, len(model.W), dtype=np.float32)
        model.b = np.float32(0.125)
        features = np.asarray([[0, 2, 4], [1, 3, 6]], dtype=np.int32)

        original_v = model.V.copy()
        original_w = model.W.copy()
        original_b = np.float32(model.b)
        original_t = model.t

        expected_embeddings = original_v[features]
        expected_sum = expected_embeddings.sum(axis=1)
        expected_logits = (
            original_b
            + original_w[features].sum(axis=1)
            + 0.5
            * (
                np.einsum("ij,ij->i", expected_sum, expected_sum)
                - np.einsum("ijk,ijk->i", expected_embeddings, expected_embeddings)
            )
        )

        first_logits, first_embeddings, first_sum = model.logits(features)
        second_logits, second_embeddings, second_sum = model.logits(features)

        np.testing.assert_allclose(first_logits, expected_logits, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(second_logits, expected_logits, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(first_embeddings, expected_embeddings)
        np.testing.assert_array_equal(second_embeddings, expected_embeddings)
        np.testing.assert_array_equal(first_sum, expected_sum)
        np.testing.assert_array_equal(second_sum, expected_sum)
        np.testing.assert_array_equal(model.V, original_v)
        np.testing.assert_array_equal(model.W, original_w)
        self.assertEqual(model.b, original_b)
        self.assertEqual(model.t, original_t)

    def test_returned_intermediates_do_not_share_memory_with_parameters(self):
        model = OfficialFM(dim=6, k=2, seed=7)
        features = np.asarray([[0, 1, 2]], dtype=np.int32)
        original_v = model.V.copy()

        _, embeddings, summed = model.logits(features)
        embeddings[:] = 99
        summed[:] = 99

        np.testing.assert_array_equal(model.V, original_v)


if __name__ == "__main__":
    unittest.main()

