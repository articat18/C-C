"""Run the constrained Gemini proposal workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.proposal import GeminiProposalClient
from experiment_engine.orchestrator import Phase3Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, help="JSON diagnostics/registry context")
    parser.add_argument(
        "--execute", action="store_true",
        help="execute the validated proposal (trains and uses Vertex AI)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    context = {}
    if args.context:
        context = json.loads(args.context.read_text(encoding="utf-8"))
    proposal = GeminiProposalClient().propose(context)
    payload = {
        "template": proposal.template,
        "hypothesis": proposal.hypothesis,
        "parameters": proposal.parameters,
        "seed": proposal.seed,
    }
    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = Phase3Orchestrator().run_proposal(proposal, verbose=not args.quiet)
    print(json.dumps({"proposal": payload, "execution": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
