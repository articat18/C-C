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

from agent.proposal import GeminiProposalClient
from agent.context import build_agent_context
from experiment_engine.orchestrator import ResearchOrchestrator
from experiment_engine.experiment_runner import ExperimentTimeout
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
            try:
                proposal = self.client.propose({**context, "status": status})
            except (RuntimeError, ValueError) as exc:
                decision = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "blocked",
                    "reason": "invalid_or_unavailable_proposal",
                    "recovery": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "action": "route_to_review",
                        "retry_count": 1,
                    },
                    "agent_wall_seconds": round(time.monotonic() - step_started, 6),
                }
                decisions.append(decision)
                _append_decision(decision)
                break
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
                },
                "proposal_provenance": dict(proposal.provenance),
            }
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


def _append_decision(decision: dict[str, Any]) -> None:
    path = resolve_editable_path("runs/agent-decisions.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, help="optional context JSON; defaults to live project context")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-continue", action="store_true")
    args = parser.parse_args()
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
