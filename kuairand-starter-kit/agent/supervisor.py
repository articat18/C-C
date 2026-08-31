"""Restart-safe bounded supervisor for governed autonomous campaigns.

The supervisor owns campaign progress, while the controller still owns the
immutable experiment package and registry.  It deliberately cannot approve or
finalize a test evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Iterator

from agent.autonomous import AutonomousResearchAgent
from agent.context import build_agent_context
from agent.proposal import ExperimentProposal
from experiment_engine.approval import ApprovalError, _require_campaign_finalist
from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.controller import ExperimentController
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.orchestrator import ResearchOrchestrator
from experiment_engine.registry import RegistryError
from experiment_boundary import resolve_editable_path


STATE_PATH = Path("runs/supervisor-state.json")
EVENTS_PATH = Path("runs/supervisor-events.jsonl")
LEASE_PATH = Path("runs/.supervisor.lock")


class SupervisorError(RuntimeError):
    """Raised when another live supervisor owns the campaign."""


class CampaignSupervisor:
    def __init__(self, *, agent: AutonomousResearchAgent | None = None, controller=None) -> None:
        self.controller = controller or ExperimentController()
        self.agent = agent or AutonomousResearchAgent(
            orchestrator=ResearchOrchestrator(controller=self.controller)
        )
        # A supplied agent owns the orchestrator used for execution; status and
        # reconciliation always use the supplied controller when one is given.
        if agent is not None:
            self.controller = agent.orchestrator.controller

    def status(self) -> dict[str, Any]:
        state = _read_json(_path(STATE_PATH), _initial_state())
        state["controller"] = self.controller.status()
        state["active_lease"] = _read_json(_path(LEASE_PATH), None)
        state["finalist"] = self._qualified_finalist()
        return state

    def run(self, *, max_steps: int = 50, execute: bool = False) -> dict[str, Any]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        with self._lease() as lease:
            state = _read_json(_path(STATE_PATH), _initial_state())
            state.update({"status": "running", "run_id": lease["run_id"], "updated_at": _utc_now()})
            self._save_state(state)
            recovered = self._reconcile_interrupted_runs()
            for proposal in recovered:
                self._heartbeat(lease, state, "retry_interrupted")
                self.agent.orchestrator.run_proposal(proposal, verbose=False, recovery_of=proposal.provenance["recovery_of"])
                self._event("interrupted_retry_completed", experiment_id=proposal.provenance["recovery_of"])

            finalist = self._qualified_finalist()
            if finalist is not None:
                return self._terminal(state, lease, "awaiting_human_approval", finalist=finalist)

            for step in range(max_steps):
                self._heartbeat(lease, state, "proposal")
                decisions = self.agent.run(
                    build_agent_context(), max_steps=1, execute=execute, auto_continue=True
                )
                decision = decisions[-1] if decisions else {"status": "blocked", "reason": "no_decision"}
                self._event("agent_decision", step=step + 1, decision=decision)
                state["completed_steps"] = int(state.get("completed_steps", 0)) + 1
                state["last_decision"] = decision
                state["updated_at"] = _utc_now()
                self._save_state(state)
                finalist = self._qualified_finalist()
                if finalist is not None:
                    return self._terminal(state, lease, "awaiting_human_approval", finalist=finalist)
                if decision.get("status") == "blocked":
                    return self._terminal(state, lease, "blocked_for_review")
                if not execute:
                    return self._terminal(state, lease, "proposal_only")
                if self.controller.status()["remaining_iterations"] <= 0:
                    return self._terminal(state, lease, "budget_exhausted")
            return self._terminal(state, lease, "step_limit_reached")

    def _qualified_finalist(self) -> dict[str, Any] | None:
        for record in sorted(
            self.controller.registry.successful_records(),
            key=lambda item: float(item.get("metrics", {}).get("valid", {}).get("primary", -1)),
            reverse=True,
        ):
            try:
                _require_campaign_finalist(record)
            except (ApprovalError, ValueError, OSError):
                continue
            return {"experiment_id": record["experiment_id"], "primary": record["metrics"]["valid"]["primary"]}
        return None

    def _reconcile_interrupted_runs(self) -> list[ExperimentProposal]:
        proposals: list[ExperimentProposal] = []
        root = _path("experiments")
        if not root.exists():
            return proposals
        for lock in root.glob("E*/.run.lock"):
            owner = _read_json(lock, {})
            if _pid_alive(owner.get("pid")):
                continue
            package = lock.parent
            experiment_id = package.name
            if (package / "result.json").exists() or (package / "failure.json").exists():
                lock.rename(package / f".run.lock.stale-{int(time.time())}")
                continue
            spec = ExperimentSpec.load(package / "spec.json")
            stale_path = package / f".run.lock.stale-{int(time.time())}"
            lock.rename(stale_path)
            failure = {
                "experiment_id": experiment_id, "status": "failed", "template": spec.template,
                "stage": spec.stage, "operator": spec.operator, "provenance": dict(spec.provenance or {}),
                "spec_fingerprint": spec.fingerprint(), "hypothesis": spec.hypothesis,
                "started_at": owner.get("started_at"), "completed_at": _utc_now(),
                "duration_seconds": 0.0, "error_type": "InterruptedRun", "error": "stale execution lease recovered by supervisor",
            }
            _atomic_json_write(package / "failure.json", failure)
            try:
                self.controller.registry.append(failure)
            except RegistryError:
                pass
            provenance = dict(spec.provenance or {})
            provenance["recovery_of"] = experiment_id
            source_ids = tuple(
                item.get("source_id", "") for item in provenance.get("research_sources", []) if isinstance(item, dict)
            )
            proposals.append(ExperimentProposal(
                template=spec.template, stage=spec.stage, operator=spec.operator,
                hypothesis=spec.hypothesis, evidence=spec.evidence, expected_effect=spec.expected_effect,
                parameters=dict(spec.parameters), seed=spec.seed, control_experiment_id=spec.control_experiment_id,
                provenance=provenance, research_source_ids=source_ids,
            ))
            self._event("stale_run_reconciled", experiment_id=experiment_id, stale_lock=stale_path.name)
        return proposals

    @contextmanager
    def _lease(self) -> Iterator[dict[str, Any]]:
        path = _path(LEASE_PATH)
        owner = _read_json(path, {}) if path.exists() else {}
        if path.exists() and _pid_alive(owner.get("pid")):
            raise SupervisorError(f"campaign is already supervised by pid {owner.get('pid')}")
        if path.exists():
            path.rename(path.with_name(f".supervisor.lock.stale-{int(time.time())}"))
        lease = {"schema_version": 1, "run_id": uuid.uuid4().hex, "pid": os.getpid(), "started_at": _utc_now(), "heartbeat_at": _utc_now()}
        _atomic_json_write(path, lease)
        self._event("supervisor_started", run_id=lease["run_id"])
        try:
            yield lease
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _heartbeat(self, lease: dict[str, Any], state: dict[str, Any], phase: str) -> None:
        lease["heartbeat_at"] = _utc_now()
        lease["phase"] = phase
        _atomic_json_write(_path(LEASE_PATH), lease)
        state["heartbeat_at"] = lease["heartbeat_at"]
        state["phase"] = phase
        self._save_state(state)

    def _terminal(self, state: dict[str, Any], lease: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
        state.update({"status": status, "updated_at": _utc_now(), **extra})
        self._save_state(state)
        self._event("supervisor_terminal", status=status, **extra)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        _atomic_json_write(_path(STATE_PATH), state)

    def _event(self, event: str, **value: Any) -> None:
        path = _path(EVENTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": _utc_now(), "event": event, **value}, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _path(path: str | Path) -> Path:
    return resolve_editable_path(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _initial_state() -> dict[str, Any]:
    return {"schema_version": 1, "status": "new", "completed_steps": 0, "updated_at": _utc_now()}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        command = subparsers.add_parser(name)
        command.add_argument("--max-steps", type=int, default=50)
        command.add_argument("--execute", action="store_true")
    subparsers.add_parser("status")
    args = parser.parse_args()
    from experiment_engine.campaign import configure_campaign
    configure_campaign(args.campaign)
    supervisor = CampaignSupervisor()
    try:
        output = supervisor.status() if args.command == "status" else supervisor.run(max_steps=args.max_steps, execute=args.execute)
    except (SupervisorError, ValueError, RegistryError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
