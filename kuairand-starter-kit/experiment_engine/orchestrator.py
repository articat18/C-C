"""Deterministic governed research orchestrator.

This is the policy layer that a future Gemini adapter can call.  It owns no
training code: all execution remains in ``ExperimentController``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from experiment_engine.controller import ControllerError, ExperimentController
from experiment_engine.phase3 import plan_phase3, run_phase3
from agent.proposal import ExperimentProposal, proposal_fingerprint


class ResearchOrchestrator:
    def __init__(self, controller: ExperimentController | None = None) -> None:
        self.controller = controller or ExperimentController()

    def run(
        self,
        *,
        max_runs: int = 1,
        seed: int = 0,
        auto_continue: bool = False,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run up to ``max_runs`` planned candidates and return an audit summary."""
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        outcomes: list[dict[str, Any]] = []
        continuations = 0
        for _ in range(max_runs):
            status = self.controller.status()
            if status["converged"]:
                if not auto_continue:
                    break
                self.controller.continue_research(
                    "Phase 3 orchestrator: current search window converged; evaluate the next approved candidate family."
                )
                continuations += 1
            paths = plan_phase3(self.controller, limit=1, seed=seed)
            if not paths:
                break
            try:
                result = run_phase3(self.controller, [paths[0]], verbose=verbose)[0]
            except ControllerError as exc:
                outcomes.append({"path": str(paths[0]), "status": "blocked", "error": str(exc)})
                break
            outcomes.append({
                "experiment_id": result["experiment_id"],
                "path": str(paths[0]),
                "status": result.get("status", "success"),
                "metrics": result.get("metrics", {}),
                "comparison": result.get("comparison", {}),
            })
        return {
            "runs": outcomes,
            "continuations": continuations,
            "status": self.controller.status(),
        }

    def run_proposal(
        self,
        proposal: ExperimentProposal,
        *,
        verbose: bool = True,
        manual_interventions: int = 0,
    ) -> dict[str, Any]:
        """Materialize one validated proposal and execute it through the controller."""
        provenance = dict(proposal.provenance)
        provenance["proposal_fingerprint"] = proposal_fingerprint(proposal)
        provenance["manual_interventions"] = manual_interventions
        spec, path = self.controller.create(
            proposal.template,
            proposal.hypothesis,
            seed=proposal.seed,
            parameters=proposal.parameters,
            stage=proposal.stage,
            operator=proposal.operator,
            evidence=proposal.evidence,
            expected_effect=proposal.expected_effect,
            provenance=provenance,
            control_experiment_id=proposal.control_experiment_id,
        )
        result = self.controller.run(spec, verbose=verbose)
        return {"spec_path": str(path), "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auto-continue", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    summary = ResearchOrchestrator().run(
        max_runs=args.max_runs,
        seed=args.seed,
        auto_continue=args.auto_continue,
        verbose=not args.quiet,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Compatibility for existing Phase 3 imports and scripts.
Phase3Orchestrator = ResearchOrchestrator
