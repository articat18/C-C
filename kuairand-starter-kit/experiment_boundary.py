"""Repository guardrails shared by the future autonomous-agent tools."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent

# These hashes establish the immutable organizer/evaluation boundary.  A planned
# update must be reviewed by a human and update the hash deliberately.
PROTECTED_FILES = {
    "official_baseline.py": "1060d22f4c67818e2e8cff4e91a0e24fbcbf65b8576089b76b793393d422bf91",
    "data.py": "d552057735b3b2bb87ac13c940beb5140d9edf6054755b9836ddb53ae1c40041",
    "evaluate.py": "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de",
    "baseline_scores.json": "950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324",
}

EDITABLE_ROOTS = (
    "analysis",
    "agent",
    "candidates",
    "experiment_engine",
    "experiments",
    "runs",
)

TRAINING_SPLIT = "train"
MODEL_SELECTION_SPLIT = "valid"
FINAL_EVALUATION_SPLIT = "test"
MAX_ITERATIONS = 50
MAX_WALL_SECONDS = 6 * 60 * 60
CONVERGENCE_EPSILON = 0.002
CONVERGENCE_PATIENCE = 3


class BoundaryViolation(RuntimeError):
    """Raised when an experiment attempts to cross a protected boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protected_files_unchanged() -> None:
    failures = []
    for relative_path, expected_hash in PROTECTED_FILES.items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing protected file: {relative_path}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            failures.append(
                f"protected file changed: {relative_path} "
                f"(expected {expected_hash}, got {actual_hash})"
            )
    if failures:
        raise BoundaryViolation("\n".join(failures))


def resolve_editable_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise BoundaryViolation(f"path is outside the repository: {candidate}") from exc

    if relative.as_posix() in PROTECTED_FILES:
        raise BoundaryViolation(f"protected file is not editable: {relative}")
    if not relative.parts or relative.parts[0] not in EDITABLE_ROOTS:
        roots = ", ".join(EDITABLE_ROOTS)
        raise BoundaryViolation(
            f"agent edits are restricted to [{roots}]; received {relative}"
        )
    return candidate


def assert_model_selection_split(split: str) -> None:
    if split != MODEL_SELECTION_SPLIT:
        raise BoundaryViolation(
            f"model selection may use only {MODEL_SELECTION_SPLIT!r}; received {split!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--can-edit", metavar="PATH")
    args = parser.parse_args()
    if not args.check and not args.can_edit:
        parser.error("choose --check or --can-edit PATH")
    if args.check:
        assert_protected_files_unchanged()
        print(f"protected boundary verified: {len(PROTECTED_FILES)} files")
    if args.can_edit:
        print(resolve_editable_path(args.can_edit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
