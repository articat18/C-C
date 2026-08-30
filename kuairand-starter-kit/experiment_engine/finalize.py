"""Human-gated final test evaluation and submission generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import baseline as baseline_models
from candidates.feature_pipeline import encode_candidate_splits
from data import load
from evaluate import evaluate
from submit import write_submission
from experiment_engine.approval import ApprovalError, require_final_approval
from experiment_engine.checkpoints import CheckpointManager, _atomic_json_write
from experiment_engine.experiment_runner import _metric_subset
from experiment_engine.experiment_spec import EXPERIMENT_ID, ExperimentSpec
from experiment_engine.experiment_templates import get_template
from experiment_engine.registry import ExperimentRegistry
from experiment_boundary import assert_protected_files_unchanged, resolve_editable_path


class FinalizationError(RuntimeError):
    """Raised when a final evaluation cannot be performed safely."""


def finalize_experiment(
    experiment_id: str,
    *,
    submission_path: str | Path | None = None,
    registry: ExperimentRegistry | None = None,
    checkpoint_manager: CheckpointManager | None = None,
) -> dict[str, Any]:
    if not EXPERIMENT_ID.fullmatch(experiment_id):
        raise FinalizationError("experiment_id must match E followed by 4-8 digits")
    assert_protected_files_unchanged()
    registry = registry or ExperimentRegistry()
    checkpoint_manager = checkpoint_manager or CheckpointManager()
    run_directory = resolve_editable_path(Path("experiments") / experiment_id)
    spec_path = run_directory / "spec.json"
    if not spec_path.is_file():
        raise FinalizationError(f"canonical experiment specification is missing: {spec_path}")
    spec = ExperimentSpec.load(spec_path)
    approval = require_final_approval(spec, registry=registry)
    result_path = run_directory / "final-result.json"
    if result_path.exists():
        raise FinalizationError(f"experiment has already been finalized: {result_path}")

    if submission_path is None:
        submission = resolve_editable_path(
            Path("runs") / experiment_id / "submission.csv"
        )
    else:
        submission = resolve_editable_path(submission_path)
    if submission.exists():
        raise FinalizationError(f"submission path already exists: {submission}")
    submission.parent.mkdir(parents=True, exist_ok=True)

    splits = load(str(spec.resolved_data_dir()))
    encoded, dimension = encode_candidate_splits(
        splits, operator_name=spec.operator
    )
    X_test, labels_test, users_test = encoded["test"]
    template = get_template(spec.template)
    member_predictions = []
    for member in range(template.ensemble_members):
        checkpoint = checkpoint_manager.load_member(experiment_id, member)
        state = checkpoint["state"]
        metadata = checkpoint["metadata"]
        if metadata.get("spec_fingerprint") != spec.fingerprint():
            raise FinalizationError(
                f"checkpoint member {member} does not match the approved specification"
            )
        model = baseline_models.FM(
            dimension,
            k=int(spec.parameters["embedding_dim"]),
            lr=float(spec.parameters["learning_rate"]),
            l2=float(spec.parameters["l2"]),
            seed=spec.seed + member,
        )
        if state["V"].shape != model.V.shape or state["W"].shape != model.W.shape:
            raise FinalizationError(f"checkpoint member {member} has incompatible shapes")
        model.V = state["V"]
        model.W = state["W"]
        model.b = np.float32(state["b"])
        member_predictions.append(model.predict(X_test))

    scores = baseline_models._standardize(np.mean(member_predictions, axis=0))
    popularity_weight = float(spec.parameters["popularity_weight"])
    if popularity_weight:
        popularity = baseline_models._standardize(
            baseline_models._pop_scores(splits["train"], splits["test"])
        )
        scores = (1.0 - popularity_weight) * scores + popularity_weight * popularity
    metrics = _metric_subset(evaluate(users_test, labels_test, scores))
    _atomic_submission_write(submission, splits["test"], scores)
    result = {
        "experiment_id": experiment_id,
        "spec_fingerprint": spec.fingerprint(),
        "status": "finalized",
        "stage": spec.stage,
        "operator": spec.operator,
        "approval": {
            "approved_by": approval["approved_by"],
            "approved_at": approval["approved_at"],
        },
        "metrics": {"test": metrics},
        "submission": submission.as_posix(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_write(result_path, result)
    return result


def _atomic_submission_write(path: Path, rows: list[Any], scores: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_submission(temporary, rows, scores)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--submission",
        help="output CSV under runs/, experiments/, or another editable root",
    )
    args = parser.parse_args()
    try:
        result = finalize_experiment(
            args.experiment_id, submission_path=args.submission
        )
    except (ApprovalError, FinalizationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
