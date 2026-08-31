from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent.autonomous import AutonomousResearchAgent
from agent.proposal import parse_proposal
from experiment_engine.experiment_runner import ExperimentTimeout


class AutonomousAgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.log = Path(self.temporary.name) / "decisions.jsonl"
        self.proposal = parse_proposal({
            "template": "bpr_hybrid",
            "stage": "features",
            "operator": "user_activity_bucket",
            "hypothesis": "User activity may separate ranking regimes.",
            "evidence": "Validation diagnostics differ by user activity.",
            "expected_effect": "Improve ranking for activity subgroups.",
            "parameters": {},
            "seed": 0,
        })

    def tearDown(self):
        self.temporary.cleanup()

    def test_convergence_stops_and_is_recorded(self):
        orchestrator = mock.Mock()
        orchestrator.controller.status.return_value = {"converged": True}
        agent = AutonomousResearchAgent(orchestrator=orchestrator, client=mock.Mock())
        with mock.patch("agent.autonomous.resolve_editable_path", return_value=self.log):
            decisions = agent.run({}, max_steps=2)
        self.assertEqual(decisions[0]["reason"], "converged")
        self.assertTrue(self.log.is_file())
        orchestrator.run_proposal.assert_not_called()

    def test_training_code_error_routes_to_review_without_retry(self):
        orchestrator = mock.Mock()
        orchestrator.controller.status.return_value = {"converged": False}
        orchestrator.run_proposal.side_effect = ValueError("candidate bug")
        client = mock.Mock()
        client.propose.return_value = self.proposal
        agent = AutonomousResearchAgent(orchestrator=orchestrator, client=client)
        with mock.patch("agent.autonomous.resolve_editable_path", return_value=self.log):
            decisions = agent.run({}, execute=True)
        self.assertEqual(decisions[0]["status"], "blocked")
        self.assertEqual(decisions[0]["recovery"]["action"], "route_to_review")
        orchestrator.run_proposal.assert_called_once()

    def test_timeout_retries_once_and_reflects_on_success(self):
        orchestrator = mock.Mock()
        orchestrator.controller.status.return_value = {"converged": False}
        orchestrator.run_proposal.side_effect = [
            ExperimentTimeout("slow"),
            {
                "result": {
                    "operator": "user_activity_bucket",
                    "comparison": {
                        "decision": "reject_or_refine",
                        "candidate": 0.60,
                        "improvement": -0.0016,
                    },
                    "diagnostics": {"validation_subgroups": {}},
                }
            },
        ]
        client = mock.Mock()
        client.propose.return_value = self.proposal
        agent = AutonomousResearchAgent(orchestrator=orchestrator, client=client)
        with mock.patch("agent.autonomous.resolve_editable_path", return_value=self.log), mock.patch(
            "agent.autonomous.build_agent_context", return_value={}
        ):
            decisions = agent.run({}, execute=True)
        self.assertEqual(orchestrator.run_proposal.call_count, 2)
        self.assertEqual(decisions[0]["recovery"]["action"], "retry_once")
        self.assertEqual(decisions[0]["reflection"]["decision"], "change_direction")

    def test_invalid_model_proposal_routes_to_source_backed_fallback(self):
        orchestrator = mock.Mock()
        orchestrator.controller.status.return_value = {"converged": False}
        orchestrator.run_proposal.return_value = {
            "result": {
                "operator": "date_period_bucket",
                "comparison": {"decision": "reject_or_refine", "candidate": 0.60},
                "diagnostics": {"validation_subgroups": {}},
            }
        }
        client = mock.Mock()
        client.propose.side_effect = ValueError("multiple scalar changes")
        agent = AutonomousResearchAgent(orchestrator=orchestrator, client=client)
        context = {
            "research_sources": [{"source_id": "S-0123456789abcdef"}],
            "experiments": [],
        }
        with mock.patch("agent.autonomous.resolve_editable_path", return_value=self.log), mock.patch(
            "agent.autonomous.build_agent_context", return_value=context
        ):
            decisions = agent.run(context, execute=True)
        self.assertEqual(decisions[0]["recovery"]["action"], "deterministic_fallback")
        self.assertEqual(decisions[0]["proposal"]["operator"], "date_period_bucket")
        orchestrator.run_proposal.assert_called_once()

    def test_transient_proposal_service_retries_with_bounded_backoff(self):
        orchestrator = mock.Mock()
        orchestrator.controller.status.return_value = {"converged": False}
        client = mock.Mock()
        client.propose.side_effect = [
            RuntimeError("503 service unavailable"),
            RuntimeError("temporary network timeout"),
            self.proposal,
        ]
        agent = AutonomousResearchAgent(orchestrator=orchestrator, client=client)
        with mock.patch("agent.autonomous.time.sleep") as sleep:
            proposal, recovery = agent._propose_with_recovery({})
        self.assertEqual(proposal, self.proposal)
        self.assertIsNone(recovery)
        self.assertEqual(client.propose.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 30])


if __name__ == "__main__":
    unittest.main()
