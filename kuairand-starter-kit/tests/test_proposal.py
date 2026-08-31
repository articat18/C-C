import json
from pathlib import Path
import tempfile
import unittest

from agent.proposal import (
    load_proposal_artifact,
    parse_proposal,
    proposal_fingerprint,
    save_proposal_artifact,
)
from experiment_boundary import REPOSITORY_ROOT


class ProposalTests(unittest.TestCase):
    def valid_proposal(self):
        return parse_proposal({
            "template": "bpr_hybrid",
            "stage": "loss",
            "operator": "none",
            "hypothesis": "Try a lower BPR weight.",
            "evidence": "Phase 3 showed a GAUC and nDCG tradeoff.",
            "expected_effect": "A lower BPR weight preserves more GAUC.",
            "parameters": {"bpr_weight": 0.5},
            "seed": 1,
        })

    def test_valid_proposal_is_normalized(self):
        proposal = self.valid_proposal()
        self.assertEqual(proposal.template, "bpr_hybrid")
        self.assertEqual(proposal.parameters["bpr_weight"], 0.5)
        self.assertEqual(proposal.parameters["embedding_dim"], 16)
        self.assertEqual(proposal.stage, "loss")
        self.assertEqual(proposal.operator, "none")

    def test_proposal_artifact_round_trip_and_tamper_rejection(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="proposal-test-", dir=runs) as directory:
            path = Path(directory) / "proposal.json"
            original = self.valid_proposal()
            save_proposal_artifact(path, original)
            loaded = load_proposal_artifact(path)
            self.assertEqual(proposal_fingerprint(loaded), proposal_fingerprint(original))

            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["proposal"]["hypothesis"] = "Modified after review."
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_proposal_artifact(path)

    def test_feature_proposal_cannot_change_model_parameter(self):
        with self.assertRaises(ValueError):
            parse_proposal({
                "template": "bpr_hybrid",
                "stage": "features",
                "operator": "video_popularity_bucket",
                "hypothesis": "Add popularity.",
                "evidence": "Popularity subgroups differ.",
                "expected_effect": "Improve cold-item ordering.",
                "parameters": {"embedding_dim": 32},
                "seed": 1,
            })

    def test_arbitrary_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_proposal({
                "template": "bpr_hybrid",
                "stage": "training",
                "operator": "none",
                "hypothesis": "unsafe",
                "evidence": "unsafe",
                "expected_effect": "unsafe",
                "parameters": {"command": "rm -rf"},
            })


if __name__ == "__main__":
    unittest.main()
