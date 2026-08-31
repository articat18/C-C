import os
from pathlib import Path
import unittest
from unittest import mock

from agent.candidate_patch import (
    CandidatePatchProposal,
    auto_promote_candidate_patch,
)
from agent.proposal import parse_proposal, proposal_to_dict
from experiment_engine.approval import ApprovalError, _require_campaign_finalist
from experiment_engine.controller import ControllerError, ExperimentController
from experiment_engine.campaign import active_campaign, campaign_scope
from experiment_boundary import resolve_editable_path


class CampaignScopeTests(unittest.TestCase):
    def test_campaign_scopes_only_runtime_artifacts(self):
        original = os.environ.get("AUTOML_CAMPAIGN")
        with campaign_scope("phase6-test"):
            self.assertEqual(active_campaign(), "phase6-test")
            self.assertIn(
                "/campaigns/phase6-test/experiments/E0001/spec.json",
                resolve_editable_path("experiments/E0001/spec.json").as_posix(),
            )
            self.assertIn(
                "/campaigns/phase6-test/runs/research/S-abc.json",
                resolve_editable_path("runs/research/S-abc.json").as_posix(),
            )
            self.assertNotIn(
                "/campaigns/phase6-test/",
                resolve_editable_path("candidates/feature_pipeline.py").as_posix(),
            )
        self.assertEqual(os.environ.get("AUTOML_CAMPAIGN"), original)

    def test_phase6_source_ids_survive_proposal_serialization(self):
        proposal = parse_proposal({
            "template": "pointwise_fm",
            "stage": "features",
            "operator": "date_period_bucket",
            "hypothesis": "Time buckets model drift.",
            "evidence": "Validation drift is measurable.",
            "expected_effect": "Improve temporal calibration.",
            "parameters": {},
            "research_source_ids": ["S-1234567890abcdef"],
        })
        self.assertEqual(
            proposal_to_dict(proposal)["research_source_ids"],
            ["S-1234567890abcdef"],
        )

    def test_auto_promotion_requires_a_campaign(self):
        proposal = CandidatePatchProposal("rationale", "feature_operator", "diff")
        with self.assertRaisesRegex(Exception, "only inside a campaign"):
            with campaign_scope(None):
                auto_promote_candidate_patch(proposal, {})
        with campaign_scope("phase6-test"), mock.patch(
            "agent.candidate_patch._promote_candidate_patch",
            return_value={"status": "promoted"},
        ) as promote:
            self.assertEqual(auto_promote_candidate_patch(proposal, {}), {"status": "promoted"})
        self.assertEqual(promote.call_args.kwargs["manual_interventions"], 0)

    def test_campaign_reservation_requires_initialization(self):
        with campaign_scope("phase6-test"), mock.patch(
            "experiment_engine.campaign.campaign_root", return_value=Path("/private/tmp/no-campaign")
        ), self.assertRaisesRegex(ControllerError, "initialized"):
            ExperimentController().create("pointwise_fm", "Controlled campaign run.")

    def test_campaign_finalist_requires_the_configured_gain(self):
        with self.assertRaisesRegex(ApprovalError, "at least 0.002"):
            _require_campaign_finalist({"metrics": {"valid": {"primary": 0.6035}}})


if __name__ == "__main__":
    unittest.main()
