import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent.supervisor import CampaignSupervisor, SupervisorError
from experiment_engine.experiment_spec import ExperimentSpec


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path_patch = mock.patch(
            "agent.supervisor._path", side_effect=lambda value: self.root / Path(value)
        )
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.temporary.cleanup()

    def test_qualified_finalist_pauses_before_agent_execution(self):
        record = {
            "experiment_id": "E0014",
            "status": "success",
            "metrics": {"valid": {"primary": 0.6038}},
        }
        controller = mock.Mock()
        controller.registry.successful_records.return_value = [record]
        agent = mock.Mock()
        agent.orchestrator.controller = controller
        supervisor = CampaignSupervisor(agent=agent)
        with mock.patch("agent.supervisor._require_campaign_finalist"):
            output = supervisor.run(execute=True)
        self.assertEqual(output["status"], "awaiting_human_approval")
        self.assertEqual(output["finalist"]["experiment_id"], "E0014")
        agent.run.assert_not_called()

    def test_live_supervisor_lease_is_never_stolen(self):
        lease = self.root / "runs" / ".supervisor.lock"
        lease.parent.mkdir(parents=True)
        lease.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        supervisor = CampaignSupervisor(controller=mock.Mock())
        with self.assertRaises(SupervisorError):
            supervisor.run()

    def test_stale_experiment_lock_becomes_failed_record_and_one_retry(self):
        package = self.root / "experiments" / "E0001"
        package.mkdir(parents=True)
        spec = ExperimentSpec.from_mapping({
            "schema_version": 2, "experiment_id": "E0001", "template": "pointwise_fm",
            "stage": "features", "operator": "date_period_bucket", "hypothesis": "Temporal signal.",
            "evidence": "Saved evidence.", "expected_effect": "Improve validation.", "parameters": {},
            "provenance": {
                "proposal_fingerprint": "a" * 64,
                "context_fingerprint": None,
                "source_model": "test",
                "token_usage": {},
                "manual_interventions": 0,
                "research_sources": [{
                    "source_id": "S-0123456789abcdef", "url": "https://example.com/source",
                    "content_sha256": "b" * 64,
                }],
            },
        })
        (package / "spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
        (package / ".run.lock").write_text(json.dumps({"pid": 999999999, "started_at": "x"}), encoding="utf-8")
        controller = mock.Mock()
        controller.registry.append = mock.Mock()
        agent = mock.Mock()
        agent.orchestrator.controller = controller
        supervisor = CampaignSupervisor(agent=agent)
        proposals = supervisor._reconcile_interrupted_runs()
        self.assertTrue((package / "failure.json").is_file())
        self.assertEqual(controller.registry.append.call_args.args[0]["error_type"], "InterruptedRun")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].provenance["recovery_of"], "E0001")


if __name__ == "__main__":
    unittest.main()
