import json
from pathlib import Path
import tempfile
import unittest

from agent.audit import build_audit_summary
from experiment_engine.registry import ExperimentRegistry
from experiment_boundary import REPOSITORY_ROOT


class AuditSummaryTests(unittest.TestCase):
    def test_summary_aggregates_resources_and_recovery(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="audit-test-", dir=runs) as directory:
            root = Path(directory)
            registry = ExperimentRegistry(root / "index.jsonl")
            provenance = {
                "proposal_fingerprint": "a" * 64,
                "token_usage": {"input_tokens": 10, "output_tokens": 2},
                "manual_interventions": 1,
            }
            registry.append({
                "experiment_id": "E0001",
                "status": "failed",
                "provenance": provenance,
                "duration_seconds": 2.0,
            })
            registry.append({
                "experiment_id": "E0002",
                "status": "success",
                "provenance": provenance,
                "duration_seconds": 3.0,
                "resources": {"training_seconds": 3.0, "gpu_hours": 0.0},
            })
            decisions = root / "decisions.jsonl"
            decisions.write_text(json.dumps({
                "agent_wall_seconds": 1.5,
                "reflection": {"decision": "change_direction"},
            }) + "\n", encoding="utf-8")
            summary = build_audit_summary(
                registry=registry,
                decisions_path=decisions,
            )

        self.assertEqual(summary["counts"]["total"], 2)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["manual_interventions"], 2)
        self.assertEqual(summary["resources"]["training_seconds"], 5.0)
        self.assertEqual(summary["resources"]["token_usage"]["input_tokens"], 20)
        self.assertEqual(summary["recoveries"][0]["recovered_experiment"], "E0002")
        self.assertEqual(summary["reflections"][0]["decision"], "change_direction")
        self.assertFalse(summary["test_accessed"])


if __name__ == "__main__":
    unittest.main()
