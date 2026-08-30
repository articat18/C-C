"""Leakage-safe historical feature builder for future candidate experiments."""

from __future__ import annotations

import collections
from typing import Mapping, Sequence

import numpy as np


Row = tuple[int, str, str, str, str, float, int]
FEATURE_FIELDS = (
    "user_activity", "video_popularity", "author_popularity", "user_tab",
    "user_author", "video_tab", "user_duration",
)


def add_history_features(
    splits: Mapping[str, Sequence[Row]],
) -> dict[str, list[tuple[object, ...]]]:
    """Append train-derived categorical features to each split."""

    train = splits["train"]
    user_counts = collections.Counter(row[1] for row in train)
    video_counts = collections.Counter(row[2] for row in train)
    author_counts = collections.Counter(row[3] for row in train)
    duration_edges = _edges([row[5] for row in train])
    user_edges = _edges(list(user_counts.values()))
    video_edges = _edges(list(video_counts.values()))
    author_edges = _edges(list(author_counts.values()))
    output: dict[str, list[tuple[object, ...]]] = {}
    for name, rows in splits.items():
        enriched = []
        for row in rows:
            user, video, author, tab = row[1], row[2], row[3], row[4]
            duration_bucket = _bucket(row[5], duration_edges)
            enriched.append(
                row
                + (
                    _bucket(user_counts.get(user, 0), user_edges),
                    _bucket(video_counts.get(video, 0), video_edges),
                    _bucket(author_counts.get(author, 0), author_edges),
                    f"{user}_{tab}", f"{user}_{author}",
                    f"{video}_{tab}", f"{user}_{duration_bucket}",
                )
            )
        output[name] = enriched
    return output


def _edges(values: Sequence[float], n: int = 10) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def _bucket(value: float, edges: np.ndarray) -> str:
    return str(int(np.searchsorted(edges, value)))
