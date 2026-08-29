"""Deterministic dataset and validation diagnostics for the research loop."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data import load
from evaluate import evaluate
from experiment_boundary import resolve_editable_path
from experiment_engine.checkpoints import _atomic_json_write


Row = tuple[int, str, str, str, str, float, int]


def profile_dataset(splits: Mapping[str, Sequence[Row]]) -> dict[str, Any]:
    """Return a stable, JSON-compatible profile for the supplied splits."""

    result = {
        "schema_version": 1,
        "splits": {
            name: summarize_rows(rows) for name, rows in sorted(splits.items())
        },
    }
    if "train" in splits and "valid" in splits:
        result["train_valid_drift"] = distribution_drift(
            splits["train"], splits["valid"]
        )
        result["suggested_experiment_families"] = suggest_experiment_families(result)
    return result


def summarize_rows(rows: Sequence[Row]) -> dict[str, Any]:
    dates = [int(row[0]) for row in rows]
    durations = np.asarray([float(row[5]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row[6]) for row in rows], dtype=np.int8)
    pairs = [(row[1], row[2]) for row in rows]
    authors = [row[3] for row in rows]
    tabs = [row[4] for row in rows]
    return {
        "rows": len(rows),
        "users": len({row[1] for row in rows}),
        "videos": len({row[2] for row in rows}),
        "authors": len(set(authors)),
        "positive_rows": int(labels.sum()),
        "positive_rate": _float(labels.mean() if len(labels) else 0.0),
        "duration": {
            "zero_rows": int(np.count_nonzero(durations == 0.0)),
            "min": _float(durations.min() if len(durations) else 0.0),
            "mean": _float(durations.mean() if len(durations) else 0.0),
            "median": _float(np.median(durations) if len(durations) else 0.0),
            "p95": _float(np.quantile(durations, 0.95) if len(durations) else 0.0),
            "max": _float(durations.max() if len(durations) else 0.0),
        },
        "duplicate_user_video_rows": len(pairs) - len(set(pairs)),
        "date_range": {
            "min": min(dates) if dates else None,
            "max": max(dates) if dates else None,
            "distinct": len(set(dates)),
        },
        "tab_counts": {
            str(key): value for key, value in sorted(Counter(tabs).items())
        },
        "label_counts": {
            "0": int(len(labels) - labels.sum()),
            "1": int(labels.sum()),
        },
    }


def distribution_drift(train_rows: Sequence[Row], valid_rows: Sequence[Row]) -> dict[str, Any]:
    train_users = {row[1] for row in train_rows}
    train_videos = {row[2] for row in train_rows}
    train_authors = {row[3] for row in train_rows}
    valid_labels = np.asarray([int(row[6]) for row in valid_rows], dtype=np.int8)
    train_labels = np.asarray([int(row[6]) for row in train_rows], dtype=np.int8)

    def unseen_rate(values: Iterable[str], known: set[str]) -> float:
        values = list(values)
        return _float(sum(value not in known for value in values) / len(values)) if values else 0.0

    return {
        "valid_positive_rate_delta": _float(
            (valid_labels.mean() if len(valid_labels) else 0.0)
            - (train_labels.mean() if len(train_labels) else 0.0)
        ),
        "valid_unseen_user_rate": unseen_rate((row[1] for row in valid_rows), train_users),
        "valid_unseen_video_rate": unseen_rate((row[2] for row in valid_rows), train_videos),
        "valid_unseen_author_rate": unseen_rate((row[3] for row in valid_rows), train_authors),
    }


def subgroup_metrics(
    rows: Sequence[Row], scores: Sequence[float], train_rows: Sequence[Row]
) -> dict[str, Any]:
    """Evaluate validation predictions across deterministic diagnostic buckets."""

    if len(rows) != len(scores):
        raise ValueError("rows and scores must have equal lengths")
    user_counts = Counter(row[1] for row in train_rows)
    video_counts = Counter(row[2] for row in train_rows)
    valid_dates = [row[0] for row in rows]
    date_midpoint = int(np.median(valid_dates)) if valid_dates else 0
    groups: dict[str, list[int]] = {}

    for index, row in enumerate(rows):
        groups.setdefault(
            f"user_activity/{_count_bucket(user_counts[row[1]])}", []
        ).append(index)
        groups.setdefault(
            f"item_popularity/{_count_bucket(video_counts[row[2]])}", []
        ).append(index)
        groups.setdefault(f"duration/{_duration_bucket(row[5])}", []).append(index)
        groups.setdefault(
            f"date_period/{'early' if row[0] <= date_midpoint else 'late'}", []
        ).append(index)

    output: dict[str, Any] = {}
    for name in sorted(groups):
        indices = groups[name]
        metrics = evaluate(
            [rows[i][1] for i in indices],
            [rows[i][6] for i in indices],
            [scores[i] for i in indices],
        )
        output[name] = {
            "rows": len(indices),
            "users": metrics["users"],
            "GAUC": _float(metrics["GAUC"]),
            "nDCG@5": _float(metrics["nDCG@5"]),
            "primary": _float(metrics["primary"]),
        }
    return output


def baseline_comparison(
    candidate: Mapping[str, float], baseline: Mapping[str, float]
) -> dict[str, Any]:
    """Compare candidate validation metrics against published baseline metrics."""

    metrics = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        actual = float(candidate[name])
        expected = float(baseline[name])
        metrics[name] = {
            "candidate": actual,
            "baseline": expected,
            "delta": _float(actual - expected),
        }
    return {"split": "valid", "metrics": metrics}


def suggest_experiment_families(profile: Mapping[str, Any]) -> list[dict[str, str]]:
    """Map deterministic profile signals to reviewed experiment directions."""

    suggestions: list[dict[str, str]] = []
    train = profile["splits"].get("train", {})
    drift = profile.get("train_valid_drift", {})
    if train.get("duration", {}).get("zero_rows", 0):
        suggestions.append(
            {
                "family": "missing_duration_category",
                "reason": "training data contains zero-duration rows",
            }
        )
    if train.get("duplicate_user_video_rows", 0):
        suggestions.append(
            {
                "family": "duplicate_interaction_ablation",
                "reason": "training data contains repeated user-video rows",
            }
        )
    if abs(float(drift.get("valid_positive_rate_delta", 0.0))) >= 0.01:
        suggestions.append(
            {
                "family": "temporal_calibration",
                "reason": "validation positive rate differs materially from training",
            }
        )
    if float(drift.get("valid_unseen_user_rate", 0.0)) >= 0.01:
        suggestions.append(
            {
                "family": "cold_user_prior",
                "reason": "validation contains a material unseen-user fraction",
            }
        )
    if float(drift.get("valid_unseen_video_rate", 0.0)) >= 0.01:
        suggestions.append(
            {
                "family": "cold_item_prior",
                "reason": "validation contains a material unseen-video fraction",
            }
        )
    if not suggestions:
        suggestions.append(
            {
                "family": "loss_or_sampling_search",
                "reason": "no deterministic data-quality trigger exceeded its threshold",
            }
        )
    return suggestions


def _count_bucket(count: int) -> str:
    if count <= 5:
        return "01-05"
    if count <= 20:
        return "06-20"
    if count <= 100:
        return "21-100"
    return "101+"


def _duration_bucket(duration: float) -> str:
    if duration == 0:
        return "missing_zero"
    if duration <= 5000:
        return "short_0-5000"
    if duration <= 15000:
        return "medium_5001-15000"
    return "long_15001+"


def _float(value: Any) -> float:
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile",))
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--output", default="analysis/dataset-profile.json")
    args = parser.parse_args()
    if args.command == "profile":
        splits = load(args.data_dir, split_names=("train", "valid"))
        report = profile_dataset(splits)
        output = resolve_editable_path(Path(args.output))
        _atomic_json_write(output, report)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
