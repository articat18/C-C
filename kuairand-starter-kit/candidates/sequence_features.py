"""Causal user-history context for the bounded sequence ranker."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from candidates.feature_pipeline import Splits, encode_candidate_splits


HISTORY_FIELD = "prior_positive_video"
NONE_TOKEN = "__NO_PRIOR_POSITIVE_VIDEO__"
UNKNOWN_TOKEN = "__UNKNOWN_PRIOR_POSITIVE_VIDEO__"


def encode_sequence_splits(
    splits: Splits,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int]:
    """Append a train-fitted, strictly-prior positive-video context field.

    Training rows at a date share history from earlier dates only.  Validation
    and test contexts are frozen from train history, so their outcomes never
    enter feature construction.
    """
    encoded, base_dimension = encode_candidate_splits(splits, operator_name="none")
    train_contexts, final_history = _training_contexts(splits["train"])
    vocabulary = {NONE_TOKEN: 0}
    for value in train_contexts:
        if value not in vocabulary:
            vocabulary[value] = len(vocabulary)
    unknown = len(vocabulary)
    output = {}
    for split_name, rows in splits.items():
        contexts = train_contexts if split_name == "train" else [
            final_history.get(row[1], NONE_TOKEN) for row in rows
        ]
        X, labels, users = encoded[split_name]
        column = np.asarray(
            [base_dimension + vocabulary.get(value, unknown) for value in contexts],
            dtype=np.int32,
        ).reshape(-1, 1)
        output[split_name] = (np.concatenate((X, column), axis=1), labels, users)
    return output, base_dimension + len(vocabulary) + 1


def _training_contexts(rows: Sequence[tuple[int, str, str, str, str, float, int]]) -> tuple[list[str], dict[str, str]]:
    contexts = [NONE_TOKEN] * len(rows)
    by_date: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_date[int(row[0])].append(index)
    history: dict[str, str] = {}
    for date in sorted(by_date):
        indices = by_date[date]
        for index in indices:
            contexts[index] = history.get(rows[index][1], NONE_TOKEN)
        # Commit labels only after every row at this date received its context.
        for index in indices:
            row = rows[index]
            if int(row[6]) == 1:
                history[row[1]] = row[2]
    return contexts, history
