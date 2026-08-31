"""Reproduce the official pointwise FM on the public validation split."""

from __future__ import annotations

import argparse
import json
from typing import Any

from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_spec import _resolve_repository_path
from experiment_boundary import REPOSITORY_ROOT, assert_protected_files_unchanged, resolve_editable_path
from data import load
from official_baseline import run_official_fm


DEFAULT_TOLERANCE = 0.002
SEED = 0


def compare_scores(
    observed: dict[str, float],
    expected: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    checks = []
    passed = True
    for metric in ("GAUC", "nDCG@5", "primary"):
        actual = float(observed[metric])
        target = float(expected[metric])
        delta = actual - target
        within_tolerance = abs(delta) <= tolerance
        passed = passed and within_tolerance
        checks.append(
            {
                "split": "valid",
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
    splits = load(
        str(_resolve_repository_path(data_dir)),
        split_names=("train", "valid"),
    )
    observed = run_official_fm(
        splits,
        k=int(config["k"]),
        lr=float(config["lr"]),
        epochs=int(config["max_epochs"]),
        batch_size=int(config["batch"]),
        patience=int(config["patience"]),
        seed=SEED,
        verbose=verbose,
    )
    expected = published["scores"]["fm_official"]["valid"]
    comparison = compare_scores(observed["valid"], expected)
    return {
        "command": "official_fm_baseline_reproduction",
        "implementation": "official_pointwise_fm",
        "published_score_key": "fm_official",
        "evaluation_split": "valid",
        "test_accessed": False,
        "seed": SEED,
        "configuration": config,
        "observed": {
            "valid": {
                metric: float(observed["valid"][metric])
                for metric in ("GAUC", "nDCG@5", "primary")
            }
        },
        "expected": {
            "valid": {
                metric: float(expected[metric])
                for metric in ("GAUC", "nDCG@5", "primary")
            }
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
