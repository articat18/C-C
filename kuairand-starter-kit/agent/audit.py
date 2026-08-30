"""Build a derived Phase 4 audit summary from append-only run evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.registry import ExperimentRegistry
from experiment_boundary import REPOSITORY_ROOT, resolve_editable_path


def build_audit_summary(
    *,
    registry: ExperimentRegistry | None = None,
    decisions_path: str | Path = "runs/agent-decisions.jsonl",
) -> dict[str, Any]:
    records = list((registry or ExperimentRegistry()).records())
    decisions = _jsonl(resolve_editable_path(decisions_path))
    experiments = []
    token_usage: dict[str, int] = {}
    manual_interventions = 0
    training_seconds = 0.0
    gpu_hours = 0.0
    revisions: dict[str, str | None] = {}
    failed_by_proposal: dict[str, list[str]] = {}
    recoveries = []
    for record in records:
        provenance = record.get("provenance", {})
        for name, amount in provenance.get("token_usage", {}).items():
            if isinstance(amount, int) and not isinstance(amount, bool):
                token_usage[name] = token_usage.get(name, 0) + amount
        manual_interventions += int(provenance.get("manual_interventions", 0))
        resources = dict(record.get("resources", {}))
        result = _result_for(record)
        if result:
            resources = dict(result.get("resources", resources))
        training_seconds += float(resources.get("training_seconds", record.get("duration_seconds", 0.0)))
        gpu_hours += float(resources.get("gpu_hours", 0.0))
        environment = result.get("environment", {}) if result else {}
        revision = environment.get("git_revision")
        if isinstance(revision, str) and revision not in revisions:
            revisions[revision] = environment.get("code_diff_hash") or _commit_diff_hash(revision)
        fingerprint = provenance.get("proposal_fingerprint")
        if record.get("status") == "failed" and isinstance(fingerprint, str):
            failed_by_proposal.setdefault(fingerprint, []).append(record["experiment_id"])
        if record.get("status") == "success" and fingerprint in failed_by_proposal:
            recoveries.append({
                "proposal_fingerprint": fingerprint,
                "failed_experiments": failed_by_proposal[fingerprint],
                "recovered_experiment": record["experiment_id"],
                "action": "repair_then_retry_same_reviewed_proposal",
            })
        experiments.append({
            "experiment_id": record.get("experiment_id"),
            "status": record.get("status"),
            "stage": record.get("stage"),
            "operator": record.get("operator"),
            "hypothesis": record.get("hypothesis"),
            "spec_fingerprint": record.get("spec_fingerprint"),
            "proposal_fingerprint": fingerprint,
            "comparison": record.get("comparison", {}),
            "duration_seconds": record.get("duration_seconds"),
            "resources": resources,
            "git_revision": revision,
        })
    explicit_recoveries = [
        decision["recovery"]
        for decision in decisions
        if isinstance(decision.get("recovery"), dict)
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 4,
        "selection_split": "valid",
        "test_accessed": False,
        "experiments": experiments,
        "counts": {
            "total": len(records),
            "successful": sum(record.get("status") == "success" for record in records),
            "failed": sum(record.get("status") == "failed" for record in records),
            "agent_decisions": len(decisions),
            "manual_interventions": manual_interventions,
        },
        "resources": {
            "training_seconds": round(training_seconds, 6),
            "gpu_hours": round(gpu_hours, 6),
            "agent_wall_seconds": round(sum(
                float(item.get("agent_wall_seconds", 0.0)) for item in decisions
            ), 6),
            "token_usage": token_usage,
        },
        "code_evidence": [
            {"git_revision": revision, "code_diff_hash": diff_hash}
            for revision, diff_hash in sorted(revisions.items())
        ],
        "recoveries": recoveries + explicit_recoveries,
        "reflections": [
            decision["reflection"]
            for decision in decisions
            if isinstance(decision.get("reflection"), dict)
        ],
    }


def save_audit_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    destination = resolve_editable_path(path)
    _atomic_json_write(destination, summary)
    return destination


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _result_for(record: dict[str, Any]) -> dict[str, Any]:
    path = record.get("result_path")
    if not isinstance(path, str):
        return {}
    result_path = REPOSITORY_ROOT / path
    if not result_path.is_file():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _commit_diff_hash(revision: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "show", "--format=", "--binary", revision],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        return hashlib.sha256(completed.stdout).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/phase4-audit-summary.json"))
    args = parser.parse_args()
    summary = build_audit_summary()
    path = save_audit_summary(args.output, summary)
    print(json.dumps({"path": path.as_posix(), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
