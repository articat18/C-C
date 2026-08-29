"""Budgeted controller for the deterministic experiment runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
from typing import Any, Iterator

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_runner import ExperimentTimeout, run_experiment
from experiment_engine.experiment_spec import ExperimentSpec, SpecificationError
from experiment_engine.experiment_templates import TemplateValidationError
from experiment_engine.registry import ExperimentRegistry, RegistryError
from experiment_boundary import (
    CONVERGENCE_EPSILON,
    CONVERGENCE_PATIENCE,
    MAX_ITERATIONS,
    assert_protected_files_unchanged,
    resolve_editable_path,
)


class ControllerError(RuntimeError):
    """Raised when controller policy prevents an experiment from running."""


class ExperimentController:
    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def run(self, spec: ExperimentSpec, *, verbose: bool = True) -> dict[str, Any]:
        self._preflight(spec)
        run_directory = resolve_editable_path(Path("experiments") / spec.experiment_id)
        if run_directory.exists():
            raise ControllerError(f"experiment directory already exists: {run_directory}")
        run_directory.mkdir(parents=True)
        _atomic_json_write(run_directory / "spec.json", spec.to_dict())

        started_at = _utc_now()
        started = time.monotonic()
        try:
            with _wall_time_limit(spec.budget.max_wall_seconds):
                result = run_experiment(spec, verbose=verbose)
            comparison = self._compare_with_history(result)
            result["comparison"] = comparison
            _atomic_json_write(run_directory / "result.json", result)
            record = {
                "experiment_id": spec.experiment_id,
                "status": "success",
                "template": spec.template,
                "spec_fingerprint": spec.fingerprint(),
                "hypothesis": spec.hypothesis,
                "started_at": started_at,
                "completed_at": result["completed_at"],
                "duration_seconds": result["duration_seconds"],
                "metrics": result["metrics"],
                "comparison": comparison,
                "result_path": (run_directory / "result.json").relative_to(
                    resolve_editable_path("experiments").parent
                ).as_posix(),
            }
            self.registry.append(record)
            return result
        except Exception as exc:
            failure = {
                "experiment_id": spec.experiment_id,
                "status": "failed",
                "template": spec.template,
                "spec_fingerprint": spec.fingerprint(),
                "hypothesis": spec.hypothesis,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "duration_seconds": round(time.monotonic() - started, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _atomic_json_write(run_directory / "failure.json", failure)
            try:
                self.registry.append(failure)
            except RegistryError:
                pass
            raise

    def status(self) -> dict[str, Any]:
        successes = self.registry.successful_records()
        all_records = list(self.registry.records())
        primary_scores = [
            float(record["metrics"]["valid"]["primary"])
            for record in successes
        ]
        return {
            "iterations": len(all_records),
            "successful": len(successes),
            "failed": len(all_records) - len(successes),
            "best_valid_primary": max(primary_scores) if primary_scores else None,
            "converged": _has_converged(primary_scores),
            "remaining_iterations": max(0, MAX_ITERATIONS - len(all_records)),
        }

    def _preflight(self, spec: ExperimentSpec) -> None:
        assert_protected_files_unchanged()
        records = list(self.registry.records())
        if len(records) >= MAX_ITERATIONS:
            raise ControllerError(f"maximum of {MAX_ITERATIONS} experiments reached")
        if any(record.get("experiment_id") == spec.experiment_id for record in records):
            raise ControllerError(
                f"experiment_id is already registered: {spec.experiment_id}"
            )
        scores = [
            float(record["metrics"]["valid"]["primary"])
            for record in records
            if record.get("status") == "success"
        ]
        if _has_converged(scores):
            raise ControllerError(
                "experiment loop has converged; human approval is required to continue"
            )

    def _compare_with_history(self, result: dict[str, Any]) -> dict[str, Any]:
        previous = self.registry.successful_records()
        score = float(result["metrics"]["valid"]["primary"])
        previous_best = max(
            (
                float(record["metrics"]["valid"]["primary"])
                for record in previous
            ),
            default=None,
        )
        improvement = None if previous_best is None else score - previous_best
        return {
            "previous_best": previous_best,
            "improvement": improvement,
            "epsilon": CONVERGENCE_EPSILON,
            "decision": (
                "keep"
                if previous_best is None or improvement > CONVERGENCE_EPSILON
                else "reject_or_refine"
            ),
        }


def _has_converged(scores: list[float]) -> bool:
    if len(scores) <= CONVERGENCE_PATIENCE:
        return False
    best = scores[0]
    weak_iterations = 0
    for score in scores[1:]:
        improvement = score - best
        if improvement > CONVERGENCE_EPSILON:
            best = score
            weak_iterations = 0
        else:
            best = max(best, score)
            weak_iterations += 1
    return weak_iterations >= CONVERGENCE_PATIENCE


@contextmanager
def _wall_time_limit(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def timeout_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise ExperimentTimeout("experiment exceeded max_wall_seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one approved experiment")
    run_parser.add_argument("spec", help="path to a JSON experiment specification")
    run_parser.add_argument("--quiet", action="store_true")
    subparsers.add_parser("status", help="show experiment-loop status")
    args = parser.parse_args()

    controller = ExperimentController()
    try:
        if args.command == "run":
            spec_path = resolve_editable_path(args.spec)
            spec = ExperimentSpec.load(spec_path)
            output = controller.run(spec, verbose=not args.quiet)
        else:
            output = controller.status()
    except (
        ControllerError,
        RegistryError,
        SpecificationError,
        TemplateValidationError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
