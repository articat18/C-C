"""Bounded Gemini-driven governed research loop.

Each model suggestion is validated before execution and every decision is
written to a local append-only audit log.  Execution is opt-in via ``--execute``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from agent.proposal import (
    ExperimentProposal, GeminiProposalClient, context_fingerprint,
    proposal_fingerprint, proposal_to_dict,
)
from agent.context import build_agent_context
from experiment_engine.orchestrator import ResearchOrchestrator
from experiment_engine.experiment_runner import ExperimentTimeout
from experiment_engine.controller import ControllerError
from experiment_boundary import resolve_editable_path


class AutonomousResearchAgent:
    def __init__(self, *, orchestrator=None, client=None) -> None:
        self.orchestrator = orchestrator or ResearchOrchestrator()
        self.client = client or GeminiProposalClient()

    def run(
        self,
        context: dict[str, Any],
        *,
        max_steps: int = 1,
        execute: bool = False,
        auto_continue: bool = False,
    ) -> list[dict[str, Any]]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        decisions = []
        for _ in range(max_steps):
            step_started = time.monotonic()
            status = self.orchestrator.controller.status()
            if status["converged"]:
                if not auto_continue:
                    decision = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "blocked",
                        "reason": "converged",
                        "agent_wall_seconds": round(time.monotonic() - step_started, 6),
                    }
                    decisions.append(decision)
                    _append_decision(decision)
                    break
                self.orchestrator.controller.continue_research(
                    "Autonomous Phase 4 agent: select a new approved direction after convergence."
                )
            proposal_recovery = None
            proposal, proposal_recovery = self._propose_with_recovery(
                {**context, "status": status}
            )
            if proposal is None:
                proposal_recovery.pop("exception")
                proposal = _fallback_proposal(context)
                if proposal is None:
                    decision = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "blocked",
                        "reason": "invalid_or_unavailable_proposal",
                        "recovery": {
                            "error_type": proposal_recovery["error_type"],
                            "error": proposal_recovery["error"],
                            "action": "route_to_review",
                            "retry_count": proposal_recovery["retry_count"],
                        },
                        "agent_wall_seconds": round(time.monotonic() - step_started, 6),
                    }
                    decisions.append(decision)
                    _append_decision(decision)
                    break
                proposal_recovery = {
                    "error_type": proposal_recovery["error_type"],
                    "error": proposal_recovery["error"],
                    "action": "deterministic_fallback",
                    "retry_count": proposal_recovery["retry_count"],
                    "selected_operator": proposal.operator,
                }
            proposal, preflight_recovery, preflight_rejections = (
                self._preflight_or_recover(proposal, context)
            )
            if proposal is None:
                decision = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "blocked",
                    "reason": "unrecoverable_controller_contract_error",
                    "recovery": preflight_recovery,
                    "preflight_rejections": preflight_rejections,
                    "agent_wall_seconds": round(time.monotonic() - step_started, 6),
                }
                decisions.append(decision)
                _append_decision(decision)
                break
            if preflight_recovery is not None:
                proposal_recovery = preflight_recovery
            decision: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "proposal": {
                    "template": proposal.template,
                    "stage": proposal.stage,
                    "operator": proposal.operator,
                    "hypothesis": proposal.hypothesis,
                    "evidence": proposal.evidence,
                    "expected_effect": proposal.expected_effect,
                    "parameters": proposal.parameters,
                    "seed": proposal.seed,
                    "control_experiment_id": proposal.control_experiment_id,
                    "research_source_ids": list(proposal.research_source_ids),
                },
                "proposal_provenance": dict(proposal.provenance),
            }
            if proposal_recovery is not None:
                decision["recovery"] = proposal_recovery
            if preflight_rejections:
                decision["preflight_rejections"] = preflight_rejections
            if execute:
                try:
                    decision["execution"] = self.orchestrator.run_proposal(
                        proposal, verbose=False
                    )
                except ExperimentTimeout as exc:
                    decision["recovery"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "action": "retry_once",
                        "retry_count": 1,
                    }
                    try:
                        decision["execution"] = self.orchestrator.run_proposal(
                            proposal, verbose=False
                        )
                    except Exception as retry_exc:
                        decision["status"] = "blocked"
                        decision["recovery"]["retry_error_type"] = type(retry_exc).__name__
                        decision["recovery"]["retry_error"] = str(retry_exc)
                        decision["recovery"]["action"] = "route_to_review"
                except Exception as exc:
                    decision["status"] = "blocked"
                    decision["recovery"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "action": "route_to_review",
                        "retry_count": 0,
                    }
                if "execution" in decision:
                    decision["reflection"] = _reflect(decision["execution"])
            else:
                decision["status"] = "proposal_only"
            decision["agent_wall_seconds"] = round(
                time.monotonic() - step_started, 6
            )
            decisions.append(decision)
            _append_decision(decision)
            if decision.get("status") == "blocked":
                break
            context = {**build_agent_context(), "previous_decision": decision}
        return decisions

    def _propose_with_recovery(
        self, context: dict[str, Any]
    ) -> tuple[ExperimentProposal | None, dict[str, Any] | None]:
        """Bound transient proposal-service outages before deterministic fallback."""
        for attempt, delay in enumerate((0, 5, 30, 120), start=0):
            if delay:
                time.sleep(delay)
            try:
                return self.client.propose(context), None
            except (RuntimeError, ValueError) as exc:
                transient = _is_transient_proposal_error(exc)
                if transient and attempt < 3:
                    continue
                return None, {
                    "exception": exc,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_count": attempt,
                    "failure_class": "transient_proposal_service" if transient else "invalid_proposal",
                }
        raise AssertionError("proposal retry loop must return")

    def _preflight_or_recover(
        self,
        proposal: ExperimentProposal,
        context: dict[str, Any],
    ) -> tuple[ExperimentProposal | None, dict[str, Any] | None, list[dict[str, Any]]]:
        """Recover one known controller-contract rejection before reservation."""
        try:
            self.orchestrator.preflight_proposal(proposal)
            return proposal, None, []
        except ControllerError as exc:
            if not _is_recoverable_contract_error(exc):
                return None, {
                    "error_type": type(exc).__name__, "error": str(exc),
                    "action": "route_to_review", "retry_count": 0,
                }, [{"proposal_fingerprint": proposal_fingerprint(proposal), "error": str(exc)}]
            contract_error = exc

        rejections = [{
            "proposal_fingerprint": proposal_fingerprint(proposal),
            "proposal": proposal_to_dict(proposal),
            "error_type": type(contract_error).__name__,
            "error": str(contract_error),
        }]
        corrected, correction_failure = self._propose_with_recovery({
            **context,
            "controller_preflight_error": str(contract_error),
            "rejected_proposal": proposal_to_dict(proposal),
            "recovery_instruction": (
                "Correct the rejected proposal once. Keep immutable matched-control "
                "settings identical, or omit the control and propose one valid change."
            ),
        })
        if corrected is not None and proposal_fingerprint(corrected) != proposal_fingerprint(proposal):
            try:
                self.orchestrator.preflight_proposal(corrected)
                return corrected, {
                    "action": "controller_contract_correction", "retry_count": 1,
                    "error_type": type(contract_error).__name__, "error": str(contract_error),
                }, rejections
            except ControllerError as correction_exc:
                rejections.append({
                    "proposal_fingerprint": proposal_fingerprint(corrected),
                    "proposal": proposal_to_dict(corrected),
                    "error_type": type(correction_exc).__name__, "error": str(correction_exc),
                })
        elif correction_failure is not None:
            rejections.append({
                "proposal_fingerprint": None,
                "error_type": correction_failure["error_type"],
                "error": correction_failure["error"],
            })

        fallback = _fallback_proposal(context)
        if fallback is not None:
            try:
                self.orchestrator.preflight_proposal(fallback)
                return fallback, {
                    "action": "deterministic_fallback_after_preflight", "retry_count": 1,
                    "error_type": type(contract_error).__name__, "error": str(contract_error),
                    "selected_operator": fallback.operator,
                }, rejections
            except ControllerError as fallback_exc:
                rejections.append({
                    "proposal_fingerprint": proposal_fingerprint(fallback),
                    "proposal": proposal_to_dict(fallback),
                    "error_type": type(fallback_exc).__name__, "error": str(fallback_exc),
                })
        return None, {
            "action": "route_to_review", "retry_count": 1,
            "error_type": type(contract_error).__name__, "error": str(contract_error),
        }, rejections


def _reflect(execution: dict[str, Any]) -> dict[str, Any]:
    result = execution["result"]
    comparison = result.get("comparison", {})
    controller_decision = comparison.get("decision", "reject_or_refine")
    matched = result.get("matched_comparison", {})
    if controller_decision == "keep":
        outcome = "keep"
    elif matched.get("decision") == "promising":
        outcome = "refine"
    elif result.get("operator", "none") != "none":
        outcome = "change_direction"
    else:
        outcome = "refine"
    subgroups = result.get("diagnostics", {}).get("validation_subgroups", {})
    weakest = sorted(
        (
            {"name": name, "primary": metrics.get("primary")}
            for name, metrics in subgroups.items()
            if isinstance(metrics.get("primary"), (int, float))
        ),
        key=lambda item: item["primary"],
    )[:3]
    return {
        "decision": outcome,
        "candidate_primary": comparison.get("candidate"),
        "improvement": comparison.get("improvement"),
        "matched_improvement": matched.get("improvement"),
        "matched_decision": matched.get("decision"),
        "weakest_validation_subgroups": weakest,
        "test_accessed": False,
    }


def _fallback_proposal(context: dict[str, Any]) -> ExperimentProposal | None:
    """Route around an unavailable/invalid model proposal with one safe screen.

    The fallback is deliberately finite and catalog-only: it never invents a
    model, changes multiple parameters, or proceeds without recorded evidence.
    """
    sources = context.get("research_sources", [])
    source_id = next(
        (item.get("source_id") for item in sources if isinstance(item, dict)
         and isinstance(item.get("source_id"), str)),
        None,
    )
    if source_id is None:
        return None
    used = {
        item.get("operator")
        for item in context.get("experiments", [])
        if isinstance(item, dict) and item.get("status") == "success"
    }
    for operator in (
        "date_period_bucket",
        "video_popularity_bucket",
        "smoothed_video_long_view_rate",
        "user_activity_bucket",
        "author_popularity_bucket",
        "user_tab_affinity",
        "user_author_affinity",
        "video_tab_affinity",
    ):
        if operator in used:
            continue
        return ExperimentProposal(
            template="pointwise_fm",
            stage="features",
            operator=operator,
            hypothesis=(
                f"Screen {operator} after the proposal service returned an invalid "
                "or unavailable response."
            ),
            evidence=(
                "Use the saved research source and validation diagnostics; this "
                "deterministic recovery changes one training-fitted feature only."
            ),
            expected_effect="Measure whether the isolated feature improves validation ranking.",
            parameters={},
            provenance={
                "context_fingerprint": context_fingerprint(context),
                "source_model": "deterministic_recovery",
                "token_usage": {},
            },
            research_source_ids=(source_id,),
        )
    return None


def _is_transient_proposal_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in (
        "timeout", "timed out", "connection", "network", "temporar",
        "unavailable", "429", "rate limit", "500", "502", "503", "504",
    ))


def _is_recoverable_contract_error(exc: ControllerError) -> bool:
    message = str(exc).lower()
    return any(token in message for token in (
        "matched control", "matched controls", "requires a matched pointwise control",
    ))


def _append_decision(decision: dict[str, Any]) -> None:
    path = resolve_editable_path("runs/agent-decisions.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="use an isolated campaign workspace")
    parser.add_argument("--context", type=Path, help="optional context JSON; defaults to live project context")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-continue", action="store_true")
    args = parser.parse_args()
    if args.campaign:
        from experiment_engine.campaign import configure_campaign
        configure_campaign(args.campaign)
    context = (
        json.loads(args.context.read_text(encoding="utf-8"))
        if args.context
        else build_agent_context()
    )
    decisions = AutonomousResearchAgent().run(
        context,
        max_steps=args.max_steps,
        execute=args.execute,
        auto_continue=args.auto_continue,
    )
    print(json.dumps(decisions, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Compatibility for callers using the Phase 3 class name.
AutonomousPhase3Agent = AutonomousResearchAgent
