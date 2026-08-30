"""Budgeted controller for the deterministic experiment runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from typing import Any, Iterator, Mapping

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_runner import ExperimentTimeout, run_experiment
from experiment_engine.experiment_spec import ExperimentSpec, SpecificationError
from experiment_engine.experiment_templates import TEMPLATES, TemplateValidationError
from experiment_engine.reference_baseline import load_baseline_reference
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


CONTINUATION_PATH = Path("experiments/research_windows.jsonl")


class ExperimentController:
    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()
        self.baseline = load_baseline_reference()

    def run(self, spec: ExperimentSpec, *, verbose: bool = True) -> dict[str, Any]:
        self._preflight(spec)
        run_directory = resolve_editable_path(Path("experiments") / spec.experiment_id)
        result_path = run_directory / "result.json"
        failure_path = run_directory / "failure.json"
        if result_path.exists() or failure_path.exists():
            raise ControllerError(f"experiment has already been executed: {run_directory}")
        run_directory.mkdir(parents=True, exist_ok=True)
        lock_path = run_directory / ".run.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
        except FileExistsError as exc:
            raise ControllerError(f"experiment is already running: {spec.experiment_id}") from exc
        spec_path = run_directory / "spec.json"
        try:
            if spec_path.exists():
                    canonical_spec = ExperimentSpec.load(spec_path)
                    if canonical_spec.fingerprint() != spec.fingerprint():
                        raise ControllerError(
                            f"submitted specification differs from canonical package: {spec_path}"
                        )
            else:
                _atomic_json_write(spec_path, spec.to_dict())

            started_at = _utc_now()
            started = time.monotonic()
            try:
                    with _wall_time_limit(spec.budget.max_wall_seconds):
                        result = run_experiment(spec, verbose=verbose)
                    comparison = self._compare_with_history(result)
                    result["comparison"] = comparison
                    _atomic_json_write(result_path, result)
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
                        "diagnostics": result.get("diagnostics", {}),
                        "comparison": comparison,
                        "result_path": result_path.relative_to(
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
                _atomic_json_write(failure_path, failure)
                try:
                    self.registry.append(failure)
                except RegistryError:
                    pass
                raise
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def create(
        self,
        template: str,
        hypothesis: str,
        *,
        seed: int = 0,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[ExperimentSpec, Path]:
        """Reserve an ID and write a validated template specification."""

        experiments_root = resolve_editable_path("experiments")
        experiments_root.mkdir(parents=True, exist_ok=True)
        used_ids = {
            str(record.get("experiment_id"))
            for record in self.registry.records()
            if record.get("experiment_id")
        }
        for path in experiments_root.rglob("*.json"):
            try:
                with path.open(encoding="utf-8") as stream:
                    value = json.load(stream)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("experiment_id"), str):
                used_ids.add(value["experiment_id"])
        used_ids.update(
            path.name
            for path in experiments_root.iterdir()
            if path.is_dir() and path.name.startswith("E")
        )

        for number in range(1, 100_000_000):
            experiment_id = f"E{number:04d}"
            if experiment_id in used_ids:
                continue
            spec = ExperimentSpec.from_mapping(
                {
                    "schema_version": 1,
                    "experiment_id": experiment_id,
                    "template": template,
                    "seed": seed,
                    "hypothesis": hypothesis,
                    "parameters": dict(parameters or {}),
                }
            )
            run_directory = experiments_root / experiment_id
            path = run_directory / "spec.json"
            try:
                run_directory.mkdir()
                _exclusive_json_write(path, spec.to_dict())
            except FileExistsError:
                used_ids.add(experiment_id)
                continue
            except Exception:
                try:
                    run_directory.rmdir()
                except OSError:
                    pass
                raise
            return spec, path
        raise ControllerError("no experiment IDs remain")

    def status(self) -> dict[str, Any]:
        successes = self.registry.successful_records()
        all_records = list(self.registry.records())
        primary_scores = [
            float(record["metrics"]["valid"]["primary"])
            for record in successes
        ]
        best_candidate = max(
            successes,
            key=lambda record: float(record["metrics"]["valid"]["primary"]),
            default=None,
        )
        best_candidate_primary = (
            float(best_candidate["metrics"]["valid"]["primary"])
            if best_candidate is not None
            else None
        )
        if best_candidate_primary is not None and best_candidate_primary > self.baseline.primary:
            best_source = best_candidate["experiment_id"]
            best_primary = best_candidate_primary
        else:
            best_source = self.baseline.name
            best_primary = self.baseline.primary
        records_since_resume = self._records_since_resume(all_records)
        return {
            "iterations": len(all_records),
            "successful": len(successes),
            "failed": len(all_records) - len(successes),
            "baseline_valid_primary": self.baseline.primary,
            "best_candidate_primary": best_candidate_primary,
            "best_candidate_improvement": (
                best_candidate_primary - self.baseline.primary
                if best_candidate_primary is not None
                else None
            ),
            "best_candidate_decision": (
                "keep"
                if best_candidate_primary is not None
                and best_candidate_primary - self.baseline.primary
                > CONVERGENCE_EPSILON
                else "reject_or_refine" if best_candidate_primary is not None else None
            ),
            "best_valid_primary": best_primary,
            "best_source": best_source,
            "converged": _has_converged([
                self.baseline.primary,
                *[float(r["metrics"]["valid"]["primary"])
                  for r in records_since_resume
                  if r.get("status") == "success"],
            ]),
            "research_window_experiments": len(records_since_resume),
            "remaining_iterations": max(0, MAX_ITERATIONS - len(all_records)),
        }

    def continue_research(self, reason: str) -> dict[str, Any]:
        """Authorize a fresh convergence window after a human-reviewed reason."""
        reason = reason.strip()
        if not reason:
            raise ControllerError("a continuation reason is required")
        records = list(self.registry.records())
        if not _has_converged([
            self.baseline.primary,
            *[float(r["metrics"]["valid"]["primary"])
              for r in self._records_since_resume(records)
              if r.get("status") == "success"],
        ]):
            raise ControllerError("the current research window has not converged")
        if len(records) >= MAX_ITERATIONS:
            raise ControllerError(f"maximum of {MAX_ITERATIONS} experiments reached")
        entry = {
            "event": "research_continuation",
            "reason": reason,
            "experiment_count": len(records),
            "timestamp": _utc_now(),
        }
        path = resolve_editable_path(CONTINUATION_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def _records_since_resume(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        path = resolve_editable_path(CONTINUATION_PATH)
        if not path.exists():
            return records
        latest: dict[str, Any] | None = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    latest = json.loads(line)
        start = int(latest.get("experiment_count", 0)) if latest else 0
        return records[start:]

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
            for record in self._records_since_resume(records)
            if record.get("status") == "success"
        ]
        if _has_converged([self.baseline.primary, *scores]):
            raise ControllerError(
                "experiment loop has converged; human approval is required to continue"
            )

    def _compare_with_history(self, result: dict[str, Any]) -> dict[str, Any]:
        previous = self.registry.successful_records()
        score = float(result["metrics"]["valid"]["primary"])
        previous_best = self.baseline.primary
        reference = self.baseline.name
        for record in previous:
            candidate_score = float(record["metrics"]["valid"]["primary"])
            if candidate_score > previous_best:
                previous_best = candidate_score
                reference = record["experiment_id"]
        improvement = score - previous_best
        return {
            "reference": reference,
            "previous_best": previous_best,
            "candidate": score,
            "improvement": improvement,
            "epsilon": CONVERGENCE_EPSILON,
            "decision": (
                "keep"
                if improvement > CONVERGENCE_EPSILON
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


def _exclusive_json_write(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one approved experiment")
    run_parser.add_argument("spec", help="path to a JSON experiment specification")
    run_parser.add_argument("--quiet", action="store_true")
    create_parser = subparsers.add_parser(
        "create", help="create a spec with the next available experiment ID"
    )
    create_parser.add_argument("--template", required=True, choices=sorted(TEMPLATES))
    create_parser.add_argument("--hypothesis", required=True)
    create_parser.add_argument("--seed", type=int, default=0)
    continue_parser = subparsers.add_parser(
        "continue", help="open a new research window after convergence"
    )
    continue_parser.add_argument("--reason", required=True)
    subparsers.add_parser("status", help="show experiment-loop status")
    args = parser.parse_args()

    controller = ExperimentController()
    try:
        if args.command == "run":
            spec_path = resolve_editable_path(args.spec)
            spec = ExperimentSpec.load(spec_path)
            output = controller.run(spec, verbose=not args.quiet)
        elif args.command == "create":
            spec, path = controller.create(
                args.template, args.hypothesis, seed=args.seed
            )
            output = {
                "experiment_id": spec.experiment_id,
                "path": path.relative_to(resolve_editable_path("experiments").parent).as_posix(),
                "spec_fingerprint": spec.fingerprint(),
            }
        elif args.command == "continue":
            output = controller.continue_research(args.reason)
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
