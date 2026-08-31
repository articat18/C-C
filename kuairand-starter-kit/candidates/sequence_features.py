"""Causal user-history context for the bounded sequence ranker."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Mapping, Sequence

import numpy as np

from candidates.feature_pipeline import Splits, encode_candidate_splits


HISTORY_FIELD = "prior_positive_video"
ATTENTION_HISTORY_FIELDS = tuple(
    f"prior_positive_video_{position}" for position in range(1, 9)
)
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


def encode_attention_sequence_splits(
    splits: Splits,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int, int]:
    """Append eight strictly-causal positive-video positions for attention.

    Each position has its own field range, preserving recency position without
    allowing a label from the current date (or any validation/test label) into
    the context.  The slots are oldest-to-newest and left-padded with NONE.
    """
    encoded, base_dimension = encode_candidate_splits(splits, operator_name="none")
    train_contexts, final_history = _training_history_windows(
        splits["train"], length=len(ATTENTION_HISTORY_FIELDS)
    )
    vocabulary = {NONE_TOKEN: 0}
    for context in train_contexts:
        for value in context:
            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)
    unknown = len(vocabulary)
    width = len(vocabulary) + 1
    output = {}
    for split_name, rows in splits.items():
        contexts = train_contexts if split_name == "train" else [
            final_history.get(row[1], (NONE_TOKEN,) * len(ATTENTION_HISTORY_FIELDS))
            for row in rows
        ]
        X, labels, users = encoded[split_name]
        if contexts:
            history_columns = np.asarray(
                [
                    [base_dimension + position * width + vocabulary.get(value, unknown)
                     for position, value in enumerate(context)]
                    for context in contexts
                ],
                dtype=np.int32,
            )
        else:
            history_columns = np.empty((0, len(ATTENTION_HISTORY_FIELDS)), dtype=np.int32)
        output[split_name] = (np.concatenate((X, history_columns), axis=1), labels, users)
    return output, base_dimension + len(ATTENTION_HISTORY_FIELDS) * width, base_dimension


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


def _training_history_windows(
    rows: Sequence[tuple[int, str, str, str, str, float, int]], *, length: int
) -> tuple[list[tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Create same-date-safe fixed-length windows from positive train events."""
    contexts = [(NONE_TOKEN,) * length for _ in rows]
    by_date: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_date[int(row[0])].append(index)
    history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=length))
    for date in sorted(by_date):
        indices = by_date[date]
        for index in indices:
            values = list(history[rows[index][1]])
            contexts[index] = tuple([NONE_TOKEN] * (length - len(values)) + values)
        for index in indices:
            row = rows[index]
            if int(row[6]) == 1:
                history[row[1]].append(row[2])
    final_history = {
        user: tuple([NONE_TOKEN] * (length - len(values)) + list(values))
        for user, values in history.items()
    }
    return contexts, final_history
