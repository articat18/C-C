"""Reviewed, train-fitted pipeline operators for candidate experiments.

The official loader and encoder remain untouched. Candidate operators either
add one categorical field to the exact official encoding or provide bounded
training-only sampling weights. All learned state comes from the training split.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from data import FIELDS, encode


Row = tuple[int, str, str, str, str, float, int]
Splits = Mapping[str, Sequence[Row]]
PIPELINE_STAGES = ("cleaning", "features", "loss", "model", "training")


class FeatureOperatorError(ValueError):
    """Raised when a specification selects an unsupported operator."""


@dataclass(frozen=True)
class FeatureOperator:
    name: str
    stages: tuple[str, ...]
    field_name: str | None
    description: str


OPERATORS: dict[str, FeatureOperator] = {
    "none": FeatureOperator(
        name="none",
        stages=("loss", "model", "training"),
        field_name=None,
        description="Use the protected five-field candidate input unchanged.",
    ),
    "missing_duration_category": FeatureOperator(
        name="missing_duration_category",
        stages=("cleaning",),
        field_name="duration_missing",
        description="Distinguish zero/missing duration from observed duration.",
    ),
    "video_popularity_bucket": FeatureOperator(
        name="video_popularity_bucket",
        stages=("features",),
        field_name="video_popularity",
        description="Bucket each video's training-only impression count.",
    ),
    "inverse_duplicate_frequency": FeatureOperator(
        name="inverse_duplicate_frequency",
        stages=("cleaning",),
        field_name=None,
        description=(
            "Downweight repeated training interactions by the inverse frequency "
            "of their label-free exact feature tuple."
        ),
    ),
}


def operator_contracts() -> dict[str, dict[str, Any]]:
    """Return JSON-safe operator metadata for proposal prompts and diagnostics."""

    return {
        name: {
            "stages": list(operator.stages),
            "field_name": operator.field_name,
            "description": operator.description,
        }
        for name, operator in OPERATORS.items()
    }


def validate_pipeline_selection(stage: str, operator_name: str) -> FeatureOperator:
    if stage not in PIPELINE_STAGES:
        raise FeatureOperatorError(
            f"unsupported pipeline stage {stage!r}; choose one of {list(PIPELINE_STAGES)}"
        )
    try:
        operator = OPERATORS[operator_name]
    except KeyError as exc:
        raise FeatureOperatorError(
            f"unknown operator {operator_name!r}; choose one of {sorted(OPERATORS)}"
        ) from exc
    if stage not in operator.stages:
        raise FeatureOperatorError(
            f"operator {operator_name!r} supports stages {list(operator.stages)}, "
            f"not {stage!r}"
        )
    return operator


def encode_candidate_splits(
    splits: Splits,
    *,
    operator_name: str = "none",
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], int]:
    """Encode official fields plus one reviewed, training-fitted operator field."""

    if operator_name not in OPERATORS:
        raise FeatureOperatorError(f"unknown operator {operator_name!r}")
    if OPERATORS[operator_name].field_name is None:
        return encode(splits)

    encoded, base_dimension = encode(splits)
    state = _fit_operator(operator_name, splits["train"])
    train_values = [_operator_value(operator_name, row, state) for row in splits["train"]]
    vocabulary: dict[str, int] = {}
    for value in train_values:
        if value not in vocabulary:
            vocabulary[value] = len(vocabulary)
    unknown = len(vocabulary)

    enriched: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split_name, rows in splits.items():
        X, labels, users = encoded[split_name]
        column = np.empty((len(rows), 1), dtype=np.int32)
        for index, row in enumerate(rows):
            value = _operator_value(operator_name, row, state)
            column[index, 0] = vocabulary.get(value, unknown) + base_dimension
        enriched[split_name] = (
            np.concatenate((X, column), axis=1),
            labels,
            users,
        )
    return enriched, base_dimension + len(vocabulary) + 1


def encoded_field_names(operator_name: str) -> tuple[str, ...]:
    try:
        operator = OPERATORS[operator_name]
    except KeyError as exc:
        raise FeatureOperatorError(f"unknown operator {operator_name!r}") from exc
    return tuple(FIELDS) + (() if operator.field_name is None else (operator.field_name,))


def training_sample_weights(
    splits: Splits,
    *,
    operator_name: str = "none",
) -> np.ndarray | None:
    """Return row-aligned training weights for a reviewed sampling operator."""

    if operator_name not in OPERATORS:
        raise FeatureOperatorError(f"unknown operator {operator_name!r}")
    if operator_name != "inverse_duplicate_frequency":
        return None
    counts = collections.Counter(_duplicate_key(row) for row in splits["train"])
    return np.asarray(
        [1.0 / counts[_duplicate_key(row)] for row in splits["train"]],
        dtype=np.float32,
    )


def operator_diagnostics(
    splits: Splits,
    *,
    operator_name: str = "none",
) -> dict[str, Any]:
    """Describe the fitted operator without exposing validation or test labels."""

    if operator_name not in OPERATORS:
        raise FeatureOperatorError(f"unknown operator {operator_name!r}")
    if operator_name != "inverse_duplicate_frequency":
        return {}
    counts = collections.Counter(_duplicate_key(row) for row in splits["train"])
    duplicated = [count for count in counts.values() if count > 1]
    weights = training_sample_weights(splits, operator_name=operator_name)
    assert weights is not None
    return {
        "duplicate_key": "date,user,video,author,tab,duration",
        "unique_training_interactions": len(counts),
        "duplicate_groups": len(duplicated),
        "rows_in_duplicate_groups": sum(duplicated),
        "excess_duplicate_rows": sum(count - 1 for count in duplicated),
        "effective_training_mass": float(weights.sum()),
        "minimum_sample_weight": float(weights.min()) if weights.size else None,
        "preserves_row_count": True,
        "uses_training_split_only": True,
    }


def _fit_operator(operator_name: str, train: Sequence[Row]) -> dict[str, Any]:
    if operator_name == "missing_duration_category":
        return {}
    if operator_name == "video_popularity_bucket":
        counts = collections.Counter(row[2] for row in train)
        values = np.asarray(list(counts.values()), dtype=np.float64)
        edges = (
            np.quantile(values, np.linspace(0, 1, 11)[1:-1])
            if values.size
            else np.asarray([], dtype=np.float64)
        )
        return {"counts": counts, "edges": edges}
    raise FeatureOperatorError(f"operator {operator_name!r} cannot be encoded")


def _operator_value(operator_name: str, row: Row, state: Mapping[str, Any]) -> str:
    if operator_name == "missing_duration_category":
        return "missing" if row[5] == 0 else "observed"
    if operator_name == "video_popularity_bucket":
        count = state["counts"].get(row[2], 0)
        return str(int(np.searchsorted(state["edges"], count)))
    raise FeatureOperatorError(f"operator {operator_name!r} cannot transform rows")


def _duplicate_key(row: Row) -> tuple[object, ...]:
    """Return an exact feature tuple without consulting the outcome label."""

    return tuple(row[:-1])
