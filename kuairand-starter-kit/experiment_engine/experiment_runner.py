"""Deterministic execution of approved experiment templates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

import baseline as baseline_models
from candidates.feature_pipeline import (
    encode_candidate_splits,
    encoded_field_names,
    operator_diagnostics,
    training_sample_weights,
)
from candidates.sequence_features import ATTENTION_HISTORY_FIELDS, HISTORY_FIELD, encode_attention_sequence_splits, encode_sequence_splits
from candidates.sequence_model import fit_causal_attention, fit_sequence_mlp
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

    The loader requests only train and validation. Candidate selection therefore
    cannot inspect test labels, features, or metrics.
    """

    assert_protected_files_unchanged()
    assert_model_selection_split("valid")
    template = get_template(spec.template)
    checkpoint_manager = checkpoint_manager or CheckpointManager()
    started = time.monotonic()
    deadline = started + spec.budget.max_wall_seconds

    loaded = load(
        str(spec.resolved_data_dir()), split_names=("train", "valid")
    )
    splits = {
        "train": loaded["train"],
        "valid": loaded["valid"],
        "test": [],
    }
    del loaded

    parameters = dict(spec.parameters)
    popularity_weight = float(parameters.pop("popularity_weight", 0.0))
    encode_fn = lambda candidate_splits: encode_candidate_splits(
        candidate_splits, operator_name=spec.operator
    )
    if template.objective == "sequence_mlp":
        encode_fn = encode_sequence_splits
    elif template.objective == "causal_attention":
        encode_fn = encode_attention_sequence_splits
    sample_weights = training_sample_weights(splits, operator_name=spec.operator)
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
        common = {
            "splits": splits,
            "k": int(parameters["embedding_dim"]),
            "lr": float(parameters["learning_rate"]),
            "epochs": spec.budget.max_epochs,
            "patience": int(parameters["patience"]),
            "seed": member_seed,
            "verbose": verbose,
            "l2": float(parameters["l2"]),
            "encode_fn": encode_fn,
            "train_sample_weights": sample_weights,
        }
        if template.objective == "sequence_mlp":
            model, encoded = fit_sequence_mlp(
                splits,
                embedding_dim=int(parameters["embedding_dim"]),
                hidden_dim=int(parameters["hidden_dim"]),
                learning_rate=float(parameters["learning_rate"]),
                l2=float(parameters["l2"]),
                epochs=spec.budget.max_epochs,
                patience=int(parameters["patience"]),
                batch_size=int(parameters["batch_size"]),
                seed=member_seed,
                encode_fn=encode_fn,
                verbose=verbose,
            )
        elif template.objective == "causal_attention":
            model, encoded = fit_causal_attention(
                splits,
                embedding_dim=int(parameters["embedding_dim"]),
                hidden_dim=int(parameters["hidden_dim"]),
                learning_rate=float(parameters["learning_rate"]),
                l2=float(parameters["l2"]),
                epochs=spec.budget.max_epochs,
                patience=int(parameters["patience"]),
                batch_size=int(parameters["batch_size"]),
                seed=member_seed,
                encode_fn=encode_fn,
                verbose=verbose,
            )
        elif template.objective == "pointwise_bce":
            model, encoded = baseline_models._fit_fm_pointwise(
                **common,
                batch_size=int(parameters["batch_size"]),
            )
        elif template.objective == "lambdarank":
            if spec.control_experiment_id is None:
                raise ValueError("LambdaRank requires a matched control checkpoint")
            control_checkpoint = checkpoint_manager.load_member(
                spec.control_experiment_id, member
            )
            control_metadata = control_checkpoint["metadata"]
            if control_metadata.get("experiment_id") != spec.control_experiment_id:
                raise ValueError("matched control checkpoint metadata is inconsistent")
            if control_metadata.get("encoded_fields") != list(
                encoded_field_names(spec.operator)
            ):
                raise ValueError("matched control checkpoint uses a different feature pipeline")
            lambda_common = dict(common)
            lambda_common["lr"] = float(lambda_common["lr"]) * 0.1
            model, encoded = baseline_models._fit_fm_lambdarank(
                **lambda_common,
                initial_state=control_checkpoint["state"],
            )
        else:
            model, encoded = baseline_models._fit_fm_bpr(
                **common,
                neg_per_pos=int(parameters["negatives_per_positive"]),
                bpr_weight=float(parameters["bpr_weight"]),
                bce_weight=float(parameters["bce_weight"]),
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
                    "stage": spec.stage,
                    "operator": spec.operator,
                    "encoded_fields": (
                        list(encoded_field_names(spec.operator)) + [HISTORY_FIELD]
                        if template.objective == "sequence_mlp"
                        else (
                            list(encoded_field_names(spec.operator)) + list(ATTENTION_HISTORY_FIELDS)
                            if template.objective == "causal_attention"
                            else list(encoded_field_names(spec.operator))
                        )
                    ),
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
        "objective": template.objective,
        "objective_diagnostics": (
            {"fine_tune_learning_rate": float(spec.parameters["learning_rate"]) * 0.1}
            if template.objective == "lambdarank" else {}
        ),
        "seed": spec.seed,
        "parameters": dict(spec.parameters),
        "stage": spec.stage,
        "operator": spec.operator,
        "encoded_fields": (
            list(encoded_field_names(spec.operator)) + [HISTORY_FIELD]
            if template.objective == "sequence_mlp"
            else (
                list(encoded_field_names(spec.operator)) + list(ATTENTION_HISTORY_FIELDS)
                if template.objective == "causal_attention"
                else list(encoded_field_names(spec.operator))
            )
        ),
        "operator_diagnostics": operator_diagnostics(
            splits, operator_name=spec.operator
        ),
        "provenance": dict(spec.provenance or {}),
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
        "resources": {
            "training_seconds": round(duration, 6),
            "gpu_hours": 0.0,
            "accelerator": "cpu_numpy",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_revision": _git_revision(),
            "code_diff_hash": _git_diff_hash(),
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


def _git_diff_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "diff", "--binary", "--", "."],
            check=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return hashlib.sha256(completed.stdout).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None
