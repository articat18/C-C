import unittest

from agent.proposal import parse_proposal


class ProposalTests(unittest.TestCase):
    def test_valid_proposal_is_normalized(self):
        proposal = parse_proposal({
            "template": "bpr_hybrid",
            "stage": "loss",
            "operator": "none",
            "hypothesis": "Try a lower BPR weight.",
            "evidence": "Phase 3 showed a GAUC and nDCG tradeoff.",
            "expected_effect": "A lower BPR weight preserves more GAUC.",
            "parameters": {"bpr_weight": 0.5},
            "seed": 1,
        })
        self.assertEqual(proposal.template, "bpr_hybrid")
        self.assertEqual(proposal.parameters["bpr_weight"], 0.5)
        self.assertEqual(proposal.parameters["embedding_dim"], 16)
        self.assertEqual(proposal.stage, "loss")
        self.assertEqual(proposal.operator, "none")

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
