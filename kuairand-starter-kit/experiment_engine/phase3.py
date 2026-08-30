"""Bounded Phase 3 BPR research workflow.

The workflow only creates specifications from the closed template catalogue.  It
does not generate source code, alter protected files, or select the hidden test
split.  Use ``plan`` to reserve a reproducible one-variable-at-a-time batch and
``run`` to execute the resulting specifications through the normal controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_engine.controller import ExperimentController
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.experiment_templates import get_template


DEFAULT_SEARCH: tuple[tuple[str, str, Any], ...] = (
    ("bpr_hybrid", "bpr_weight", 0.25),
    ("bpr_hybrid", "bpr_weight", 1.5),
    ("bpr_hybrid", "learning_rate", 0.0005),
    ("bpr_hybrid", "learning_rate", 0.002),
    ("bpr_hybrid", "negatives_per_positive", 2),
    ("bpr_hybrid", "negatives_per_positive", 8),
    ("bpr_hybrid", "embedding_dim", 32),
    ("bpr_ensemble", "popularity_weight", 0.1),
)


def plan_phase3(
    controller: ExperimentController,
    *,
    limit: int = len(DEFAULT_SEARCH),
    seed: int = 0,
) -> list[Path]:
    """Reserve up to ``limit`` unique one-variable Phase 3 specifications."""
    if limit < 1:
        raise ValueError("limit must be positive")
    paths: list[Path] = []
    existing: dict[tuple[str, int, tuple[tuple[str, Any], ...]], Path] = {
        _signature(spec): path
        for path in Path("experiments").glob("E*/spec.json")
        if (spec := _load_optional(path)) is not None
    }
    for template, parameter, value in DEFAULT_SEARCH:
        if len(paths) >= limit:
            break
        hypothesis = (
            f"Phase 3: changing {parameter} to {value} improves validation ranking."
        )
        normalized = get_template(template).normalize_parameters({parameter: value})
        signature = (template, seed, tuple(sorted(normalized.items())))
        if signature in existing:
            existing_path = existing[signature]
            package = existing_path.parent
            if not (package / "result.json").exists() and not (package / "failure.json").exists():
                paths.append(existing_path)
            continue
        spec, path = controller.create(
            template,
            hypothesis,
            seed=seed,
            parameters={parameter: value},
        )
        existing[_signature(spec)] = path
        paths.append(path)
    return paths


def run_phase3(
    controller: ExperimentController, paths: list[str | Path], *, verbose: bool = True
) -> list[dict[str, Any]]:
    """Run planned specifications through the standard controller."""
    results = []
    for path in paths:
        spec = ExperimentSpec.load(path)
        results.append(controller.run(spec, verbose=verbose))
    return results


def replicate_phase3(
    controller: ExperimentController,
    source_path: str | Path,
    seeds: list[int],
) -> list[Path]:
    """Create seed replications of a successful Phase 3 specification."""
    source = ExperimentSpec.load(source_path)
    if not source.hypothesis.startswith("Phase 3:"):
        raise ValueError(f"not a Phase 3 specification: {source_path}")
    paths = []
    for seed in seeds:
        if seed == source.seed:
            continue
        hypothesis = f"{source.hypothesis} Replication seed {seed}."
        _, path = controller.create(
            source.template,
            hypothesis,
            seed=seed,
            parameters=dict(source.parameters),
        )
        paths.append(path)
    return paths


def _load_optional(path: Path) -> ExperimentSpec | None:
    try:
        return ExperimentSpec.load(path)
    except Exception:
        return None


def _signature(spec: ExperimentSpec) -> tuple[str, int, tuple[tuple[str, Any], ...]]:
    return (spec.template, spec.seed, tuple(sorted(spec.parameters.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="reserve a bounded Phase 3 batch")
    plan.add_argument("--limit", type=int, default=len(DEFAULT_SEARCH))
    plan.add_argument("--seed", type=int, default=0)
    run = subparsers.add_parser("run", help="run planned Phase 3 specifications")
    run.add_argument("spec", nargs="+", help="paths returned by plan")
    run.add_argument("--quiet", action="store_true")
    replicate = subparsers.add_parser("replicate", help="create multi-seed replications")
    replicate.add_argument("spec")
    replicate.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    controller = ExperimentController()
    if args.command == "plan":
        print(json.dumps([str(path) for path in plan_phase3(
            controller, limit=args.limit, seed=args.seed
        )], indent=2))
    elif args.command == "run":
        results = run_phase3(controller, args.spec, verbose=not args.quiet)
        print(json.dumps([
            {"experiment_id": result["experiment_id"], "metrics": result["metrics"]}
            for result in results
        ], indent=2))
    else:
        print(json.dumps([str(path) for path in replicate_phase3(
            controller, args.spec, args.seeds
        )], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
