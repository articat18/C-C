"""Run the constrained Gemini proposal workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.context import build_agent_context
from agent.proposal import (
    GeminiProposalClient,
    load_proposal_artifact,
    proposal_fingerprint,
    proposal_to_dict,
    save_proposal_artifact,
)
from experiment_engine.orchestrator import ResearchOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="use an isolated campaign workspace")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--context", type=Path, help="JSON diagnostics/registry context")
    source.add_argument(
        "--proposal", type=Path,
        help="validated proposal artifact to inspect or execute without calling Gemini",
    )
    parser.add_argument(
        "--output-proposal", type=Path,
        help="save a new validated Gemini proposal without training",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="execute the validated proposal (trains and uses Vertex AI)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.campaign:
        from experiment_engine.campaign import configure_campaign
        configure_campaign(args.campaign)
    if args.output_proposal and args.execute:
        parser.error("--output-proposal cannot be combined with --execute")
    if args.output_proposal and args.proposal:
        parser.error("--output-proposal creates a proposal and cannot use --proposal")
    try:
        if args.proposal:
            proposal = load_proposal_artifact(args.proposal)
            manual_interventions = 1
        else:
            context = (
                json.loads(args.context.read_text(encoding="utf-8"))
                if args.context
                else build_agent_context()
            )
            proposal = GeminiProposalClient().propose(context)
            manual_interventions = 0
            if args.output_proposal:
                path = save_proposal_artifact(args.output_proposal, proposal)
                print(json.dumps({
                    "path": path.as_posix(),
                    "proposal_fingerprint": proposal_fingerprint(proposal),
                    "proposal": proposal_to_dict(proposal),
                    "token_usage": proposal.provenance.get("token_usage", {}),
                }, indent=2, sort_keys=True))
                return 0
    except (ValueError, RuntimeError) as exc:
        parser.error(f"could not obtain an approved proposal: {exc}")
    payload = proposal_to_dict(proposal)
    if not args.execute:
        print(json.dumps({
            "proposal": payload,
            "proposal_fingerprint": proposal_fingerprint(proposal),
            "token_usage": proposal.provenance.get("token_usage", {}),
        }, indent=2, sort_keys=True))
        return 0
    result = ResearchOrchestrator().run_proposal(
        proposal,
        verbose=not args.quiet,
        manual_interventions=manual_interventions,
    )
    print(json.dumps({"proposal": payload, "execution": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
