"""Deterministic execution of approved experiment templates."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import subprocess
import time
from typing import Any

import numpy as np

import baseline as baseline_models
from data import load
from evaluate import evaluate
from experiment_engine.checkpoints import CheckpointManager
from experiment_engine.diagnostics import baseline_comparison, subgroup_metrics
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.experiment_templates import get_template
from experiment_boundary import assert_model_selection_split, assert_protected_files_unchanged


def _published_validation_metrics() -> dict[str, float]:
    import json
    from experiment_boundary import REPOSITORY_ROOT

    with (REPOSITORY_ROOT / "baseline_scores.json").open(encoding="utf-8") as stream:
        return {
            name: float(value)
            for name, value in json.load(stream)["scores"]["fm_official"]["valid"].items()
        }


class ExperimentTimeout(TimeoutError):
    """Raised when a run crosses its specification's wall-clock budget."""


def run_experiment(
    spec: ExperimentSpec,
    *,
    checkpoint_manager: CheckpointManager | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a validated template and return JSON-serializable validation evidence.

    The true test rows are intentionally removed before encoding.  Candidate
    selection therefore cannot inspect test labels, features, or metrics.
    """

    assert_protected_files_unchanged()
    assert_model_selection_split("valid")
    template = get_template(spec.template)
    checkpoint_manager = checkpoint_manager or CheckpointManager()
    started = time.monotonic()
    deadline = started + spec.budget.max_wall_seconds

    loaded = load(str(spec.resolved_data_dir()))
    splits = {
        "train": loaded["train"],
        "valid": loaded["valid"],
        "test": [],
    }
    del loaded

    parameters = dict(spec.parameters)
    popularity_weight = float(parameters.pop("popularity_weight"))
    member_predictions: list[np.ndarray] = []
    checkpoints = []
    member_metrics = []
    valid_labels = None
    valid_users = None

    for member in range(template.ensemble_members):
        _check_deadline(deadline)
        member_seed = spec.seed + member
        if verbose:
            print(
                f"[{spec.experiment_id}] member {member + 1}/"
                f"{template.ensemble_members}, seed={member_seed}"
            )
        model, encoded = baseline_models._fit_fm_bpr(
            splits,
            k=int(parameters["embedding_dim"]),
            lr=float(parameters["learning_rate"]),
            epochs=spec.budget.max_epochs,
            patience=int(parameters["patience"]),
            seed=member_seed,
            verbose=verbose,
            neg_per_pos=int(parameters["negatives_per_positive"]),
            bpr_weight=float(parameters["bpr_weight"]),
            bce_weight=float(parameters["bce_weight"]),
            l2=float(parameters["l2"]),
        )
        X_valid, valid_labels, valid_users = encoded["valid"]
        predictions = model.predict(X_valid)
        metrics = evaluate(valid_users, valid_labels, predictions)
        member_predictions.append(predictions)
        member_metrics.append(_metric_subset(metrics))
        checkpoints.append(
            checkpoint_manager.save_member(
                spec.experiment_id,
                member,
                model=model,
                metadata={
                    "experiment_id": spec.experiment_id,
                    "spec_fingerprint": spec.fingerprint(),
                    "template": spec.template,
                    "member": member,
                    "seed": member_seed,
                    "validation_metrics": _metric_subset(metrics),
                    "created_at": _utc_now(),
                },
            )
        )
        _check_deadline(deadline)

    assert valid_labels is not None and valid_users is not None
    scores = baseline_models._standardize(np.mean(member_predictions, axis=0))
    if popularity_weight:
        popularity = baseline_models._standardize(
            baseline_models._pop_scores(splits["train"], splits["valid"])
        )
        scores = (1.0 - popularity_weight) * scores + popularity_weight * popularity
    valid_metrics = _metric_subset(evaluate(valid_users, valid_labels, scores))
    duration = time.monotonic() - started
    _check_deadline(deadline)
    return {
        "experiment_id": spec.experiment_id,
        "spec_fingerprint": spec.fingerprint(),
        "template": spec.template,
        "status": "success",
        "selection_split": "valid",
        "metrics": {"valid": valid_metrics},
        "diagnostics": {
            "baseline_comparison": baseline_comparison(
                valid_metrics, _published_validation_metrics()
            ),
            "validation_subgroups": subgroup_metrics(
                splits["valid"], scores, splits["train"]
            ),
        },
        "member_metrics": member_metrics,
        "checkpoints": checkpoints,
        "rows": {"train": len(splits["train"]), "valid": len(splits["valid"])},
        "duration_seconds": round(duration, 6),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_revision": _git_revision(),
        },
        "completed_at": _utc_now(),
    }


def _metric_subset(metrics: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in ("GAUC", "nDCG@5", "primary", "users", "rows"):
        if name not in metrics:
            continue
        result[name] = (
            int(metrics[name]) if name in {"users", "rows"} else float(metrics[name])
        )
    return result


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise ExperimentTimeout("experiment exceeded max_wall_seconds")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
