"""Bounded Gemini-driven governed research loop.

Each model suggestion is validated before execution and every decision is
written to a local append-only audit log.  Execution is opt-in via ``--execute``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from agent.proposal import GeminiProposalClient
from agent.context import build_agent_context
from experiment_engine.orchestrator import ResearchOrchestrator
from experiment_boundary import resolve_editable_path


class AutonomousResearchAgent:
    def __init__(self) -> None:
        self.orchestrator = ResearchOrchestrator()
        self.client = GeminiProposalClient()

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
            status = self.orchestrator.controller.status()
            if status["converged"]:
                if not auto_continue:
                    decisions.append({"status": "blocked", "reason": "converged"})
                    break
                self.orchestrator.controller.continue_research(
                    "Autonomous Phase 4 agent: select a new approved direction after convergence."
                )
            proposal = self.client.propose({**context, "status": status})
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
                },
            }
            if execute:
                decision["execution"] = self.orchestrator.run_proposal(proposal, verbose=False)
            else:
                decision["status"] = "proposal_only"
            decisions.append(decision)
            context = {**context, "previous_decision": decision}
        path = resolve_editable_path("runs/agent-decisions.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for decision in decisions:
                stream.write(json.dumps(decision, sort_keys=True, default=str) + "\n")
        return decisions


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
