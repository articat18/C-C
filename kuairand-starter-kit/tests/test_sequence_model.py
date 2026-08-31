import unittest

import numpy as np

from candidates.sequence_features import NONE_TOKEN, _training_contexts
from candidates.sequence_model import CausalSequenceMLP
from experiment_engine.experiment_templates import get_template


class SequenceFeatureTests(unittest.TestCase):
    def test_training_context_never_uses_same_date_outcomes(self):
        rows = [
            (1, "u", "v1", "a", "t", 1.0, 1),
            (1, "u", "v2", "a", "t", 1.0, 1),
            (2, "u", "v3", "a", "t", 1.0, 0),
        ]
        contexts, final_history = _training_contexts(rows)
        self.assertEqual(contexts[:2], [NONE_TOKEN, NONE_TOKEN])
        self.assertEqual(contexts[2], "v2")
        self.assertEqual(final_history["u"], "v2")


class SequenceModelTests(unittest.TestCase):
    def test_sequence_ensemble_uses_three_members(self):
        self.assertEqual(get_template("sequence_ensemble").ensemble_members, 3)

    def test_model_trains_and_round_trips_checkpoint_state(self):
        model = CausalSequenceMLP(
            12, embedding_dim=4, hidden_dim=4, learning_rate=0.01, l2=1e-6, seed=7
        )
        X = np.asarray([[0, 1, 8], [0, 2, 9], [3, 1, 8], [3, 2, 9]], dtype=np.int32)
        y = np.asarray([1, 0, 1, 0], dtype=np.float32)
        before = model.predict(X)
        model.step(X, y)
        after = model.predict(X)
        self.assertFalse(np.allclose(before, after))
        state = {name: value.copy() for name, value in model.checkpoint_state().items()}
        restored = CausalSequenceMLP(
            12, embedding_dim=4, hidden_dim=4, learning_rate=0.01, l2=1e-6, seed=99
        )
        restored.restore(state)
        np.testing.assert_allclose(restored.predict(X), after)


if __name__ == "__main__":
    unittest.main()
