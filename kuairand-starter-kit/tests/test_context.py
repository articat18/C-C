import unittest

from agent.context import build_agent_context


class AgentContextTests(unittest.TestCase):
    def test_context_contains_validation_only_state(self):
        context = build_agent_context(limit=3)
        self.assertEqual(context["phase"], 4)
        self.assertEqual(context["constraints"]["selection_split"], "valid")
        self.assertFalse(context["constraints"]["test_accessed"])
        self.assertIn("status", context)
        self.assertIn("video_popularity_bucket", context["constraints"]["approved_operators"])
        self.assertIn(
            "inverse_duplicate_frequency",
            context["constraints"]["approved_operators"],
        )
        self.assertTrue(context["constraints"]["one_change_per_iteration"])
        for experiment in context["experiments"]:
            self.assertNotIn("test", experiment.get("metrics", {}))


if __name__ == "__main__":
    unittest.main()
