"""Leakage-safe historical feature builder for future candidate experiments."""

from __future__ import annotations

import collections
from typing import Mapping, Sequence

import numpy as np


Row = tuple[int, str, str, str, str, float, int]
FEATURE_FIELDS = (
    "user_activity", "video_popularity", "author_popularity", "user_tab",
    "user_author", "video_tab", "user_duration", "user_video_exposure",
    "video_recency", "date_period",
)

COUNT_FIELDS = {
    "user_activity": 1,
    "video_popularity": 2,
    "author_popularity": 3,
}


def add_history_features(
    splits: Mapping[str, Sequence[Row]],
) -> dict[str, list[tuple[object, ...]]]:
    """Append train-derived categorical features to each split."""

    states = {
        field: fit_history_feature(field, splits["train"])
        for field in FEATURE_FIELDS
    }
    output: dict[str, list[tuple[object, ...]]] = {}
    for name, rows in splits.items():
        enriched = []
        for row in rows:
            enriched.append(
                row
                + tuple(
                    history_feature_value(
                        field, row, states[field], training=name == "train"
                    )
                    for field in FEATURE_FIELDS
                )
            )
        output[name] = enriched
    return output


def fit_history_feature(field: str, train: Sequence[Row]) -> dict[str, object]:
    """Fit one reviewed history field from training rows only."""

    if field not in FEATURE_FIELDS:
        raise ValueError(f"unknown history feature: {field}")
    if field in COUNT_FIELDS:
        index = COUNT_FIELDS[field]
        counts = collections.Counter(row[index] for row in train)
        return {"counts": counts, "edges": _edges(list(counts.values()))}
    if field == "user_duration":
        return {"duration_edges": _edges([row[5] for row in train])}
    if field == "user_video_exposure":
        by_pair_date: collections.Counter[tuple[str, str, int]] = collections.Counter(
            (row[1], row[2], row[0]) for row in train
        )
        prior_counts: dict[tuple[str, str, int], int] = {}
        totals: collections.Counter[tuple[str, str]] = collections.Counter()
        for user, video, date in sorted(by_pair_date, key=lambda key: key[2]):
            pair = (user, video)
            prior_counts[(user, video, date)] = totals[pair]
            totals[pair] += by_pair_date[(user, video, date)]
        return {
            "prior_counts": prior_counts,
            "totals": totals,
            "edges": _edges(list(prior_counts.values()) + list(totals.values())),
        }
    if field == "video_recency":
        dates: dict[str, set[int]] = collections.defaultdict(set)
        for row in train:
            dates[row[2]].add(row[0])
        gaps: dict[tuple[str, int], int | None] = {}
        observed_gaps: list[int] = []
        last_dates: dict[str, int] = {}
        for video, video_dates in dates.items():
            previous = None
            for date in sorted(video_dates):
                gap = None if previous is None else date - previous
                gaps[(video, date)] = gap
                if gap is not None:
                    observed_gaps.append(gap)
                previous = date
            last_dates[video] = max(video_dates)
        return {
            "gaps": gaps,
            "last_dates": last_dates,
            "edges": _edges(observed_gaps),
        }
    if field == "date_period":
        return {"date_edges": _edges([row[0] for row in train])}
    return {}


def history_feature_value(
    field: str,
    row: Row,
    state: Mapping[str, object],
    *,
    training: bool = False,
) -> str:
    """Transform one row with a previously fitted history-field state."""

    if field in COUNT_FIELDS:
        key = row[COUNT_FIELDS[field]]
        counts = state["counts"]
        edges = state["edges"]
        assert isinstance(counts, collections.Counter)
        assert isinstance(edges, np.ndarray)
        return _bucket(counts.get(key, 0), edges)
    user, video, author, tab = row[1], row[2], row[3], row[4]
    if field == "user_tab":
        return f"{user}_{tab}"
    if field == "user_author":
        return f"{user}_{author}"
    if field == "video_tab":
        return f"{video}_{tab}"
    if field == "user_duration":
        edges = state["duration_edges"]
        assert isinstance(edges, np.ndarray)
        return f"{user}_{_bucket(row[5], edges)}"
    if field == "user_video_exposure":
        edges = state["edges"]
        assert isinstance(edges, np.ndarray)
        if training:
            prior_counts = state["prior_counts"]
            assert isinstance(prior_counts, dict)
            count = prior_counts[(user, video, row[0])]
        else:
            totals = state["totals"]
            assert isinstance(totals, collections.Counter)
            count = totals.get((user, video), 0)
        return _bucket(count, edges)
    if field == "video_recency":
        edges = state["edges"]
        assert isinstance(edges, np.ndarray)
        if training:
            gaps = state["gaps"]
            assert isinstance(gaps, dict)
            gap = gaps[(video, row[0])]
        else:
            last_dates = state["last_dates"]
            assert isinstance(last_dates, dict)
            last_date = last_dates.get(video)
            gap = None if last_date is None else max(0, row[0] - last_date)
        return "new" if gap is None else _bucket(gap, edges)
    if field == "date_period":
        edges = state["date_edges"]
        assert isinstance(edges, np.ndarray)
        return _bucket(row[0], edges)
    raise ValueError(f"unknown history feature: {field}")


def _edges(values: Sequence[float], n: int = 10) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def _bucket(value: float, edges: np.ndarray) -> str:
    return str(int(np.searchsorted(edges, value)))
