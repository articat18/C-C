import unittest

from agent.config import ConfigurationError, VertexConfig


class VertexConfigTests(unittest.TestCase):
    def test_loads_complete_vertex_configuration(self):
        config = VertexConfig.from_environment(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_CLOUD_PROJECT": "example-project",
                "GOOGLE_CLOUD_LOCATION": "global",
                "VERTEX_MODEL": "example-model",
            }
        )
        self.assertEqual(config.project, "example-project")
        self.assertEqual(config.location, "global")
        self.assertEqual(config.model, "example-model")

    def test_rejects_missing_values(self):
        with self.assertRaises(ConfigurationError):
            VertexConfig.from_environment({"GOOGLE_GENAI_USE_VERTEXAI": "true"})

    def test_rejects_non_vertex_mode(self):
        with self.assertRaises(ConfigurationError):
            VertexConfig.from_environment(
                {
                    "GOOGLE_GENAI_USE_VERTEXAI": "false",
                    "GOOGLE_CLOUD_PROJECT": "example-project",
                    "GOOGLE_CLOUD_LOCATION": "global",
                    "VERTEX_MODEL": "example-model",
                }
            )


if __name__ == "__main__":
    unittest.main()
