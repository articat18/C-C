import unittest

from candidates.history_features import FEATURE_FIELDS, add_history_features


class HistoryFeatureTests(unittest.TestCase):
    def test_features_are_train_derived_and_preserve_rows(self):
        train = [
            (20220408, "u1", "v1", "a1", "home", 1000.0, 1),
            (20220408, "u1", "v2", "a2", "home", 2000.0, 0),
        ]
        valid = [(20220422, "u9", "v9", "a9", "home", 9000.0, 0)]
        enriched = add_history_features({"train": train, "valid": valid})

        self.assertEqual(len(enriched["train"][0]), 7 + len(FEATURE_FIELDS))
        self.assertEqual(enriched["train"][0][:7], train[0])
        self.assertEqual(enriched["valid"][0][:7], valid[0])
        self.assertEqual(len(enriched["valid"][0]), 7 + len(FEATURE_FIELDS))

    def test_validation_labels_do_not_change_feature_values(self):
        train = [
            (20220408, "u1", "v1", "a1", "home", 1000.0, 1),
            (20220408, "u1", "v2", "a2", "home", 2000.0, 0),
        ]
        valid = [(20220422, "u9", "v9", "a9", "home", 9000.0, 0)]
        first = add_history_features({"train": train, "valid": valid})
        changed_valid = [valid[0][:6] + (1,)]
        second = add_history_features({"train": train, "valid": changed_valid})
        self.assertEqual(first["valid"][0][7:], second["valid"][0][7:])


if __name__ == "__main__":
    unittest.main()
