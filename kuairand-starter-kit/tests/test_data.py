import csv
from pathlib import Path
import tempfile
import unittest

from data import load


class SelectiveSplitLoadingTests(unittest.TestCase):
    def test_development_load_does_not_read_test_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_csv(
                root / "video_features_basic_pure.csv",
                ["video_id", "author_id"],
                [["v1", "a1"]],
            )
            fields = [
                "date",
                "user_id",
                "video_id",
                "tab",
                "duration_ms",
                "long_view",
            ]
            self._write_csv(
                root / "log_standard_4_08_to_4_21_pure.csv",
                fields,
                [["20220408", "u1", "v1", "1", "1000", "1"]],
            )
            # The hidden-test row deliberately has no label value. A selective
            # train/valid load must reject the date before touching long_view.
            self._write_csv(
                root / "log_standard_4_22_to_5_08_pure.csv",
                fields,
                [
                    ["20220422", "u1", "v1", "1", "1000", "0"],
                    ["20220429", "u1", "v1", "1", "1000", None],
                ],
            )

            splits = load(root, split_names=("train", "valid"))

        self.assertEqual(set(splits), {"train", "valid"})
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["valid"]), 1)

    @staticmethod
    def _write_csv(path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
