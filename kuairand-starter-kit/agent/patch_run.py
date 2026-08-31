"""Generate a fingerprinted Gemini candidate patch without applying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.candidate_patch import GeminiCandidatePatchClient, save_candidate_patch_artifact
from agent.context import build_agent_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="use an isolated campaign workspace")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--output-patch", type=Path, required=True)
    args = parser.parse_args()
    if args.campaign:
        from experiment_engine.campaign import configure_campaign
        configure_campaign(args.campaign)
    context = (
        json.loads(args.context.read_text(encoding="utf-8"))
        if args.context
        else build_agent_context()
    )
    context["candidate_sources"] = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in (
            Path("candidates/feature_pipeline.py"),
            Path("candidates/history_features.py"),
        )
    }
    proposal = GeminiCandidatePatchClient().propose(context)
    path = save_candidate_patch_artifact(args.output_patch, proposal)
    print(json.dumps({
        "path": path.as_posix(),
        "content_hash": proposal.content_hash,
        "rationale": proposal.rationale,
        "affected_contract": proposal.affected_contract,
        "token_usage": proposal.provenance.get("token_usage", {}),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
