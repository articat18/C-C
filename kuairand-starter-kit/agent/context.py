"""Build an evidence context for the Gemini research policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_engine.controller import ExperimentController
from experiment_engine.registry import ExperimentRegistry


def build_agent_context(*, limit: int = 20) -> dict[str, Any]:
    """Return recent, validation-only state suitable for a proposal prompt."""
    controller = ExperimentController()
    records = list(ExperimentRegistry().records())
    experiments = []
    for record in records[-limit:]:
        item = {
            "experiment_id": record.get("experiment_id"),
            "status": record.get("status"),
            "template": record.get("template"),
            "hypothesis": record.get("hypothesis"),
            "comparison": record.get("comparison", {}),
        }
        if record.get("status") == "success":
            item["metrics"] = {"valid": record.get("metrics", {}).get("valid", {})}
        spec_path = Path("experiments") / str(record.get("experiment_id")) / "spec.json"
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            item["seed"] = spec.get("seed")
            item["parameters"] = spec.get("parameters", {})
        experiments.append(item)
    continuation_path = Path("experiments/research_windows.jsonl")
    continuations = []
    if continuation_path.is_file():
        for line in continuation_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if line.strip():
                continuations.append(json.loads(line))
    diagnostics_path = Path("analysis/dataset-profile.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
    return {
        "phase": 3,
        "baseline": {"primary": controller.baseline.primary, "name": controller.baseline.name},
        "status": controller.status(),
        "experiments": experiments,
        "continuations": continuations,
        "diagnostics": diagnostics,
        "constraints": {
            "selection_split": "valid",
            "test_accessed": False,
            "approved_templates": ["bpr_hybrid", "bpr_ensemble"],
            "instruction": "Choose an evidence-backed direction; do not repeat exhausted BPR settings without justification.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    context = build_agent_context(limit=args.limit)
    encoded = json.dumps(context, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
