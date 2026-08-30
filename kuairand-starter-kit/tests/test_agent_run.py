from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from agent import run as agent_run
from agent.proposal import parse_proposal, save_proposal_artifact
from experiment_engine.orchestrator import ResearchOrchestrator
from experiment_boundary import REPOSITORY_ROOT


class AgentRunTests(unittest.TestCase):
    def test_saved_proposal_execution_does_not_call_gemini_again(self):
        proposal = parse_proposal({
            "template": "bpr_hybrid",
            "stage": "features",
            "operator": "video_popularity_bucket",
            "hypothesis": "Use training-only popularity.",
            "evidence": "Popularity subgroups differ on validation.",
            "expected_effect": "Improve cold-item ordering.",
            "parameters": {},
            "seed": 0,
        })
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agent-run-test-", dir=runs) as directory:
            path = Path(directory) / "proposal.json"
            save_proposal_artifact(path, proposal)
            orchestrator = mock.Mock()
            orchestrator.run_proposal.return_value = {"spec_path": "unused", "result": {}}
            argv = ["agent.run", "--proposal", str(path), "--execute", "--quiet"]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "agent.run.GeminiProposalClient"
            ) as gemini, mock.patch(
                "agent.run.ResearchOrchestrator", return_value=orchestrator
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(agent_run.main(), 0)

        gemini.assert_not_called()
        executed = orchestrator.run_proposal.call_args.args[0]
        self.assertEqual(executed.operator, "video_popularity_bucket")
        self.assertEqual(
            orchestrator.run_proposal.call_args.kwargs["manual_interventions"], 1
        )

    def test_orchestrator_binds_proposal_provenance_to_specification(self):
        proposal = parse_proposal({
            "template": "bpr_hybrid",
            "stage": "features",
            "operator": "video_popularity_bucket",
            "hypothesis": "Use training-only popularity.",
            "evidence": "Popularity subgroups differ on validation.",
            "expected_effect": "Improve cold-item ordering.",
            "parameters": {},
            "seed": 0,
        })
        controller = mock.Mock()
        specification = mock.sentinel.specification
        controller.create.return_value = (specification, Path("experiments/E0021/spec.json"))
        controller.run.return_value = {"status": "success"}

        ResearchOrchestrator(controller).run_proposal(
            proposal, verbose=False, manual_interventions=1
        )

        provenance = controller.create.call_args.kwargs["provenance"]
        self.assertEqual(provenance["manual_interventions"], 1)
        self.assertEqual(len(provenance["proposal_fingerprint"]), 64)
        controller.run.assert_called_once_with(specification, verbose=False)


if __name__ == "__main__":
    unittest.main()
