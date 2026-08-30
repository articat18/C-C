"""KuaiRand-Pure data loading, official splits, and canonical encoding."""

import csv
import os

import numpy as np

LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def load(data_dir, split_names=None):
    """Read requested official splits without materializing unrequested labels."""

    requested = tuple(SPLITS) if split_names is None else tuple(split_names)
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown splits: {', '.join(unknown)}")
    if not requested:
        raise ValueError("at least one split must be requested")

    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as stream:
        for row in csv.DictReader(stream):
            vid2author[row["video_id"]] = row["author_id"]

    output = {name: [] for name in requested}
    files = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for filename in files:
        with open(os.path.join(data_dir, filename)) as stream:
            for raw in csv.DictReader(stream):
                date = int(raw["date"])
                split = next(
                    (
                        name for name in requested
                        if SPLITS[name][0] <= date <= SPLITS[name][1]
                    ),
                    None,
                )
                if split is None:
                    continue
                output[split].append(
                    (
                        date,
                        raw["user_id"],
                        raw["video_id"],
                        vid2author.get(raw["video_id"], "UNK"),
                        raw["tab"],
                        float(raw["duration_ms"]),
                        1 if raw[LABEL] != "0" else 0,
                    )
                )
    return output


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Map the five canonical categorical fields to integer IDs."""

    train = splits["train"]
    edges = _bucket_edges([row[5] for row in train])

    def raw(row):
        return [
            row[1], row[2], row[3], row[4],
            str(int(np.searchsorted(edges, row[5]))),
        ]

    vocabs = [dict() for _ in FIELDS]
    for row in train:
        for index, value in enumerate(raw(row)):
            if value not in vocabs[index]:
                vocabs[index][value] = len(vocabs[index])
    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    encoded = {}
    for name, rows in splits.items():
        features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        labels = np.empty(len(rows), dtype=np.float32)
        users = []
        for index, row in enumerate(rows):
            for field, value in enumerate(raw(row)):
                features[index, field] = (
                    vocabs[field].get(value, unknown[field]) + offsets[field]
                )
            labels[index] = row[6]
            users.append(row[1])
        encoded[name] = (features, labels, users)
    return encoded, int(sum(dimensions))
