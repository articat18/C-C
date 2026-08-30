import unittest

from agent.proposal import parse_proposal


class ProposalTests(unittest.TestCase):
    def test_valid_proposal_is_normalized(self):
        proposal = parse_proposal({
            "template": "bpr_hybrid",
            "hypothesis": "Try a lower BPR weight.",
            "parameters": {"bpr_weight": 0.5},
            "seed": 1,
        })
        self.assertEqual(proposal.template, "bpr_hybrid")
        self.assertEqual(proposal.parameters["bpr_weight"], 0.5)
        self.assertEqual(proposal.parameters["embedding_dim"], 16)

    def test_arbitrary_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_proposal({
                "template": "bpr_hybrid",
                "hypothesis": "unsafe",
                "parameters": {"command": "rm -rf"},
            })


if __name__ == "__main__":
    unittest.main()
