"""Reproduce the stable published-score FM candidate deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_spec import _resolve_repository_path
from experiment_boundary import REPOSITORY_ROOT, assert_protected_files_unchanged, resolve_editable_path
from baseline import run_bpr_ensemble
from data import load


DEFAULT_TOLERANCE = 0.002
SEED = 0


def compare_scores(
    observed: dict[str, dict[str, float]],
    expected: dict[str, dict[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    checks = []
    passed = True
    for split in ("valid", "test"):
        for metric in ("GAUC", "nDCG@5", "primary"):
            actual = float(observed[split][metric])
            target = float(expected[split][metric])
            delta = actual - target
            within_tolerance = abs(delta) <= tolerance
            passed = passed and within_tolerance
            checks.append(
                {
                    "split": split,
                    "metric": metric,
                    "observed": actual,
                    "expected": target,
                    "delta": delta,
                    "within_tolerance": within_tolerance,
                }
            )
    return {"passed": passed, "tolerance": tolerance, "checks": checks}


def reproduce(data_dir: str, *, verbose: bool = True) -> dict[str, Any]:
    assert_protected_files_unchanged()
    with (REPOSITORY_ROOT / "baseline_scores.json").open(encoding="utf-8") as stream:
        published = json.load(stream)
    config = published["scores"]["fm_official"]["config"]
    splits = load(str(_resolve_repository_path(data_dir)))
    observed = run_bpr_ensemble(
        splits,
        k=int(config["k"]),
        lr=float(config["lr"]),
        epochs=int(config["max_epochs"]),
        bs=int(config["batch"]),
        patience=int(config["patience"]),
        seed=SEED,
        verbose=verbose,
    )
    expected = published["scores"]["fm_official"]
    comparison = compare_scores(observed, expected)
    return {
        "command": "stable_fm_baseline_reproduction",
        "implementation": "bpr_ensemble",
        "published_score_key": "fm_official",
        "seed": SEED,
        "configuration": config,
        "observed": {
            split: {
                metric: float(observed[split][metric])
                for metric in ("GAUC", "nDCG@5", "primary")
            }
            for split in ("valid", "test")
        },
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument(
        "--output",
        help="optional JSON path under experiments/, runs/, or another editable root",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = reproduce(args.data_dir, verbose=not args.quiet)
    if args.output:
        output = resolve_editable_path(args.output)
        _atomic_json_write(output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
