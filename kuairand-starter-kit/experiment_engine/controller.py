"""Budgeted controller for the deterministic experiment runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import signal
import time
import uuid
from typing import Any, Iterator, Mapping

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_runner import ExperimentTimeout, run_experiment
from experiment_engine.experiment_spec import (
    SCHEMA_VERSION,
    ExperimentSpec,
    SpecificationError,
    infer_pipeline_stage,
)
from experiment_engine.experiment_templates import TEMPLATES, TemplateValidationError
from experiment_engine.reference_baseline import load_baseline_reference
from experiment_engine.registry import ExperimentRegistry, RegistryError
from experiment_engine.campaign import active_campaign
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
        _create_run_lock(lock_path, experiment_id=spec.experiment_id)
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
                    result["global_comparison"] = comparison
                    matched_comparison = self._matched_comparison(spec, result)
                    if matched_comparison is not None:
                        result["matched_comparison"] = matched_comparison
                    _atomic_json_write(result_path, result)
                    record = {
                        "experiment_id": spec.experiment_id,
                        "status": "success",
                        "template": spec.template,
                        "stage": spec.stage,
                        "operator": spec.operator,
                        "seed": spec.seed,
                        "parameters": dict(spec.parameters),
                        "budget": {
                            "max_epochs": spec.budget.max_epochs,
                            "max_wall_seconds": spec.budget.max_wall_seconds,
                        },
                        "data_dir": spec.data_dir,
                        "control_experiment_id": spec.control_experiment_id,
                        "provenance": dict(spec.provenance or {}),
                        "spec_fingerprint": spec.fingerprint(),
                        "hypothesis": spec.hypothesis,
                        "started_at": started_at,
                        "completed_at": result["completed_at"],
                        "duration_seconds": result["duration_seconds"],
                        "resources": result.get("resources", {}),
                        "metrics": result["metrics"],
                        "diagnostics": result.get("diagnostics", {}),
                        "comparison": comparison,
                        "global_comparison": comparison,
                        **(
                            {"matched_comparison": matched_comparison}
                            if matched_comparison is not None else {}
                        ),
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
                        "stage": spec.stage,
                        "operator": spec.operator,
                        "provenance": dict(spec.provenance or {}),
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
        stage: str | None = None,
        operator: str = "none",
        evidence: str | None = None,
        expected_effect: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        control_experiment_id: str | None = None,
    ) -> tuple[ExperimentSpec, Path]:
        """Reserve an ID and write a validated template specification."""

        if active_campaign() is not None:
            from experiment_engine.campaign import campaign_root
            if not (campaign_root() / "campaign.json").is_file():
                raise ControllerError("campaign must be initialized before reserving experiments")
            from agent.research import validate_source_ids
            supplied_sources = dict(provenance or {}).get("research_sources", [])
            if not isinstance(supplied_sources, list):
                raise ControllerError("campaign provenance requires a research_sources list")
            validated_sources = validate_source_ids([
                str(item.get("source_id", "")) if isinstance(item, Mapping) else ""
                for item in supplied_sources
            ])
            provenance = {**dict(provenance or {}), "research_sources": validated_sources}

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
            selected_stage = stage or infer_pipeline_stage(template, parameters)
            spec = ExperimentSpec.from_mapping(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": experiment_id,
                    "template": template,
                    "seed": seed,
                    "hypothesis": hypothesis,
                    "parameters": dict(parameters or {}),
                    "stage": selected_stage,
                    "operator": operator,
                    "evidence": evidence or hypothesis,
                    "expected_effect": expected_effect or hypothesis,
                    **({"provenance": dict(provenance)} if provenance else {}),
                    **(
                        {"control_experiment_id": control_experiment_id}
                        if control_experiment_id else {}
                    ),
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
        self.validate_candidate(spec)
        records = list(self.registry.records())
        if any(record.get("experiment_id") == spec.experiment_id for record in records):
            raise ControllerError(
                f"experiment_id is already registered: {spec.experiment_id}"
            )

    def validate_candidate(self, spec: ExperimentSpec) -> None:
        """Validate a candidate against mutable campaign state without writing it.

        Proposal syntax is checked before this point.  This method covers the
        registry-dependent rules that are needed before reserving an ID.
        """
        assert_protected_files_unchanged()
        records = list(self.registry.records())
        if len(records) >= MAX_ITERATIONS:
            raise ControllerError(f"maximum of {MAX_ITERATIONS} experiments reached")
        self._validated_control(spec, records)
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

    def _validated_control(
        self,
        spec: ExperimentSpec,
        records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if spec.control_experiment_id is None:
            return None
        records = records if records is not None else list(self.registry.records())
        control = next(
            (
                record for record in records
                if record.get("experiment_id") == spec.control_experiment_id
            ),
            None,
        )
        if control is None or control.get("status") != "success":
            raise ControllerError(
                f"matched control is not a successful registered experiment: "
                f"{spec.control_experiment_id}"
            )
        model_comparison = (
            spec.template in {"sequence_mlp", "sequence_ensemble"}
            and control.get("template") == "pointwise_fm"
            and spec.stage == "model"
            and control.get("operator", "none") == "none"
            and spec.operator == "none"
        )
        control_parameters = dict(spec.parameters)
        if model_comparison:
            control_parameters.pop("hidden_dim", None)
        expected = {
            "seed": spec.seed,
            "parameters": control_parameters,
            "budget": {
                "max_epochs": spec.budget.max_epochs,
                "max_wall_seconds": spec.budget.max_wall_seconds,
            },
            "data_dir": spec.data_dir,
        }
        mismatches = [
            name for name, value in expected.items() if control.get(name) != value
        ]
        if mismatches:
            raise ControllerError(
                "matched control differs in immutable experiment settings: "
                + ", ".join(mismatches)
            )
        objective_comparison = (
            spec.template == "lambdarank_fm"
            and control.get("template") == "pointwise_fm"
            and spec.stage == "loss"
            and control.get("operator", "none") == spec.operator
        )
        feature_comparison = (
            control.get("template") == spec.template
            and control.get("operator", "none") == "none"
            and spec.operator != "none"
        )
        if not objective_comparison and not feature_comparison and not model_comparison:
            raise ControllerError(
                "matched controls must compare one feature against an operator-none "
                "control, a sequence model against pointwise FM, or LambdaRank against the same pointwise pipeline"
            )
        return control

    def _matched_comparison(
        self, spec: ExperimentSpec, result: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        control = self._validated_control(spec)
        if control is None:
            return None
        candidate = float(result["metrics"]["valid"]["primary"])
        reference = float(control["metrics"]["valid"]["primary"])
        improvement = candidate - reference
        evidence_threshold = 0.0005
        return {
            "reference": spec.control_experiment_id,
            "control": reference,
            "candidate": candidate,
            "improvement": improvement,
            "evidence_threshold": evidence_threshold,
            "decision": "promising" if improvement >= evidence_threshold else "reject",
        }


def _create_run_lock(lock_path: Path, *, experiment_id: str) -> None:
    """Create an exclusive, inspectable execution lease.

    A supervisor can distinguish an active owner from a lock left behind by a
    killed process; the controller itself never removes another process's lock.
    """
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "pid": os.getpid(),
        "run_id": uuid.uuid4().hex,
        "started_at": _utc_now(),
        "heartbeat_at": _utc_now(),
    }
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ControllerError(f"experiment is already running: {experiment_id}") from exc
    try:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    parser.add_argument(
        "--campaign",
        help="use an isolated campaign workspace under campaigns/<name>/",
    )
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
    create_parser.add_argument(
        "--stage", choices=("cleaning", "features", "loss", "model", "training")
    )
    create_parser.add_argument("--operator", default="none")
    create_parser.add_argument("--evidence")
    create_parser.add_argument("--expected-effect")
    create_parser.add_argument("--control-experiment-id")
    create_parser.add_argument(
        "--research-source-id", action="append", default=[],
        help="saved S-... evidence ID (required for campaign experiments)",
    )
    continue_parser = subparsers.add_parser(
        "continue", help="open a new research window after convergence"
    )
    continue_parser.add_argument("--reason", required=True)
    subparsers.add_parser("status", help="show experiment-loop status")
    subparsers.add_parser("init-campaign", help="initialize the selected Phase 6 campaign")
    args = parser.parse_args()

    try:
        if args.campaign:
            from experiment_engine.campaign import configure_campaign
            configure_campaign(args.campaign)
        if args.command == "init-campaign":
            if not args.campaign:
                raise ControllerError("init-campaign requires --campaign")
            from experiment_engine.campaign import initialize_campaign
            output = initialize_campaign(args.campaign)
            print(json.dumps(output, sort_keys=True, indent=2))
            return 0
        controller = ExperimentController()
        if args.command == "run":
            spec_path = resolve_editable_path(args.spec)
            spec = ExperimentSpec.load(spec_path)
            output = controller.run(spec, verbose=not args.quiet)
        elif args.command == "create":
            provenance = None
            if args.research_source_id:
                from agent.research import validate_source_ids
                sources = validate_source_ids(args.research_source_id)
                fingerprint_input = json.dumps({
                    "template": args.template, "hypothesis": args.hypothesis,
                    "seed": args.seed, "sources": [item["source_id"] for item in sources],
                }, sort_keys=True, separators=(",", ":")).encode("utf-8")
                provenance = {
                    "proposal_fingerprint": hashlib.sha256(fingerprint_input).hexdigest(),
                    "context_fingerprint": None,
                    "source_model": "manual_campaign_control",
                    "token_usage": {},
                    "manual_interventions": 1,
                    "research_sources": sources,
                }
            spec, path = controller.create(
                args.template,
                args.hypothesis,
                seed=args.seed,
                stage=args.stage,
                operator=args.operator,
                evidence=args.evidence,
                expected_effect=args.expected_effect,
                provenance=provenance,
                control_experiment_id=args.control_experiment_id,
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
