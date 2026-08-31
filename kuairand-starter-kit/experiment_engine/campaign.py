"""Campaign-scoped storage and immutable initialization evidence.

Campaigns reuse the deterministic engine while keeping experiment IDs, registry
records, decisions, approvals, and final artifacts physically separate.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator

from experiment_engine.checkpoints import _atomic_json_write
from experiment_boundary import (
    CONVERGENCE_EPSILON,
    MAX_ITERATIONS,
    MAX_WALL_SECONDS,
    REPOSITORY_ROOT,
    assert_protected_files_unchanged,
    resolve_editable_path,
)


CAMPAIGN_ENV = "AUTOML_CAMPAIGN"
CAMPAIGN_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")
SCOPED_ROOTS = frozenset({"analysis", "experiments", "runs"})


class CampaignError(ValueError):
    """Raised when campaign selection or initialization is invalid."""


def validate_campaign(name: str) -> str:
    if not CAMPAIGN_NAME.fullmatch(name):
        raise CampaignError("campaign must use lowercase letters, digits, and hyphens")
    return name


def active_campaign() -> str | None:
    value = os.environ.get(CAMPAIGN_ENV)
    return validate_campaign(value) if value else None


def configure_campaign(name: str | None) -> None:
    """Select a campaign for the current process, or restore legacy paths."""
    if name is None:
        os.environ.pop(CAMPAIGN_ENV, None)
    else:
        os.environ[CAMPAIGN_ENV] = validate_campaign(name)


@contextmanager
def campaign_scope(name: str | None) -> Iterator[None]:
    previous = os.environ.get(CAMPAIGN_ENV)
    configure_campaign(name)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CAMPAIGN_ENV, None)
        else:
            os.environ[CAMPAIGN_ENV] = previous


def campaign_root(name: str | None = None) -> Path:
    selected = validate_campaign(name) if name else active_campaign()
    if selected is None:
        raise CampaignError("no campaign is active")
    return REPOSITORY_ROOT / "campaigns" / selected


def scoped_path(path: str | Path) -> Path:
    """Map a campaign-relative analysis/experiments/runs path into its root."""
    candidate = Path(path)
    if candidate.is_absolute() or not candidate.parts or active_campaign() is None:
        return candidate
    if candidate.parts[0] not in SCOPED_ROOTS:
        return candidate
    return Path("campaigns") / active_campaign() / candidate


def initialize_campaign(name: str, *, phase: int = 6) -> dict[str, Any]:
    """Create the immutable campaign manifest before the first experiment."""
    name = validate_campaign(name)
    if phase != 6:
        raise CampaignError("only the Phase 6 campaign initializer is supported")
    assert_protected_files_unchanged()
    root = campaign_root(name)
    manifest = resolve_editable_path(root / "campaign.json")
    if manifest.exists():
        with manifest.open(encoding="utf-8") as stream:
            return json.load(stream)
    payload = {
        "campaign": name,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "budget": {"max_iterations": MAX_ITERATIONS, "max_wall_seconds": MAX_WALL_SECONDS},
        "promotion": {"minimum_validation_primary_delta": CONVERGENCE_EPSILON, "replication_seeds": 3},
        "test_accessed": False,
        "historical_experiments_read": False,
        "layout": {"analysis": "analysis", "experiments": "experiments", "runs": "runs"},
    }
    _atomic_json_write(manifest, payload)
    return payload
