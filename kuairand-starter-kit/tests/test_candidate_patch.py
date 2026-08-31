import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agent.candidate_patch import (
    CandidatePatchError,
    PROMOTION_CONFIRMATION,
    load_candidate_patch_artifact,
    parse_candidate_patch,
    promote_candidate_patch,
    save_candidate_patch_artifact,
    validate_candidate_patch,
)
from experiment_boundary import REPOSITORY_ROOT


SAFE_PATCH = """diff --git a/candidates/generated_probe.py b/candidates/generated_probe.py
new file mode 100644
index 0000000..fbd37cc
--- /dev/null
+++ b/candidates/generated_probe.py
@@ -0,0 +1,3 @@
+\"\"\"Safe generated candidate probe.\"\"\"
+def value():
+    return 1
"""


class CandidatePatchTests(unittest.TestCase):
    def proposal(self, patch=SAFE_PATCH):
        return parse_candidate_patch({
            "rationale": "Test a bounded candidate implementation.",
            "affected_contract": "feature_operator",
            "patch": patch,
        })

    def test_artifact_round_trip_and_tamper_rejection(self):
        runs = REPOSITORY_ROOT / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="patch-test-", dir=runs) as directory:
            path = Path(directory) / "patch.json"
            proposal = self.proposal()
            save_candidate_patch_artifact(path, proposal)
            self.assertEqual(load_candidate_patch_artifact(path), proposal)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["proposal"]["rationale"] = "tampered"
            artifact["proposal"]["patch"] += "\n"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(CandidatePatchError, "fingerprint mismatch"):
                load_candidate_patch_artifact(path)

    def test_rejects_paths_outside_candidates(self):
        unsafe = SAFE_PATCH.replace(
            "candidates/generated_probe.py", "data.py"
        )
        with self.assertRaisesRegex(CandidatePatchError, "outside candidates"):
            self.proposal(unsafe)

    def test_rejects_banned_import_without_touching_real_worktree(self):
        unsafe = SAFE_PATCH.replace(
            "def value():", "import os\n+def value():"
        )
        proposal = self.proposal(unsafe)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "candidates").mkdir()
            target = project / "candidates" / "generated_probe.py"
            target.write_text("import os\n", encoding="utf-8")
            with mock.patch(
                "agent.candidate_patch._extract_clean_checkout", return_value=project
            ), mock.patch("agent.candidate_patch._run"):
                with self.assertRaisesRegex(CandidatePatchError, "banned import"):
                    validate_candidate_patch(proposal)
        self.assertFalse((REPOSITORY_ROOT / "candidates/generated_probe.py").exists())

    def test_promotion_requires_confirmation_and_matching_report(self):
        proposal = self.proposal()
        report = {
            "status": "accepted",
            "content_hash": proposal.content_hash,
            "touched_paths": ["candidates/generated_probe.py"],
        }
        with self.assertRaisesRegex(CandidatePatchError, "explicit confirmation"):
            promote_candidate_patch(proposal, report, confirmation="no")
        changed = dict(report)
        changed["content_hash"] = "0" * 64
        with self.assertRaisesRegex(CandidatePatchError, "fingerprint mismatch"):
            promote_candidate_patch(
                proposal,
                changed,
                confirmation=PROMOTION_CONFIRMATION,
            )
        with self.assertRaisesRegex(CandidatePatchError, "sandbox result evidence"):
            promote_candidate_patch(
                proposal,
                report,
                confirmation=PROMOTION_CONFIRMATION,
            )


if __name__ == "__main__":
    unittest.main()
