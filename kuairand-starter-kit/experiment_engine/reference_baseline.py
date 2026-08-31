"""Authoritative validation reference used for candidate comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import json

from experiment_boundary import REPOSITORY_ROOT, assert_protected_files_unchanged


@dataclass(frozen=True)
class BaselineReference:
    name: str
    score_key: str
    split: str
    primary: float


def load_baseline_reference() -> BaselineReference:
    """Load the protected published validation score.

    Candidate decisions use the published validation primary rather than a
    machine-specific reproduction result or any test metric.
    """

    assert_protected_files_unchanged()
    score_key = "fm_official"
    split = "valid"
    with (REPOSITORY_ROOT / "baseline_scores.json").open(encoding="utf-8") as stream:
        published = json.load(stream)
    return BaselineReference(
        name="stable_published_baseline",
        score_key=score_key,
        split=split,
        primary=float(published["scores"][score_key][split]["primary"]),
    )

