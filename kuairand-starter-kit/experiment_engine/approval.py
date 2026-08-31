"""Human approval receipts for final test evaluation and submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from experiment_engine.experiment_spec import EXPERIMENT_ID, ExperimentSpec
from experiment_engine.registry import ExperimentRegistry
from experiment_engine.campaign import active_campaign
from experiment_engine.reference_baseline import load_baseline_reference
from experiment_boundary import assert_protected_files_unchanged, resolve_editable_path


APPROVAL_PHRASE = "I_APPROVE_FINAL_TEST_AND_SUBMISSION"


class ApprovalError(RuntimeError):
    """Raised when final evaluation lacks a valid human approval receipt."""


def grant_final_approval(
    experiment_id: str,
    *,
    approved_by: str,
    confirmation: str,
    registry: ExperimentRegistry | None = None,
) -> dict[str, Any]:
    assert_protected_files_unchanged()
    if confirmation != APPROVAL_PHRASE:
        raise ApprovalError(
            f"confirmation must exactly equal {APPROVAL_PHRASE!r}"
        )
    if not approved_by.strip():
        raise ApprovalError("approved_by must be non-empty")
    if not EXPERIMENT_ID.fullmatch(experiment_id):
        raise ApprovalError("experiment_id must match E followed by 4-8 digits")

    registry = registry or ExperimentRegistry()
    record = _successful_record(registry, experiment_id)
    if active_campaign() is not None:
        _require_campaign_finalist(record)
    spec_path = resolve_editable_path(Path("experiments") / experiment_id / "spec.json")
    if not spec_path.is_file():
        raise ApprovalError(f"canonical experiment specification is missing: {spec_path}")
    spec = ExperimentSpec.load(spec_path)
    if record.get("spec_fingerprint") != spec.fingerprint():
        raise ApprovalError("registry and canonical specification fingerprints differ")

    receipt = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "spec_fingerprint": spec.fingerprint(),
        "approved_by": approved_by.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "scope": "final_test_and_submission",
        "confirmation": APPROVAL_PHRASE,
    }
    path = approval_path(experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json_write(path, receipt)
    return receipt


def require_final_approval(
    spec: ExperimentSpec,
    *,
    registry: ExperimentRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or ExperimentRegistry()
    record = _successful_record(registry, spec.experiment_id)
    if record.get("spec_fingerprint") != spec.fingerprint():
        raise ApprovalError("registry and canonical specification fingerprints differ")
    path = approval_path(spec.experiment_id)
    if not path.is_file():
        raise ApprovalError(
            f"final approval is missing for {spec.experiment_id}; a human must run "
            "experiment_engine.approval first"
        )
    try:
        with path.open(encoding="utf-8") as stream:
            receipt = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"could not read approval receipt: {exc}") from exc
    expected = {
        "experiment_id": spec.experiment_id,
        "spec_fingerprint": spec.fingerprint(),
        "scope": "final_test_and_submission",
        "confirmation": APPROVAL_PHRASE,
    }
    mismatches = [
        key for key, value in expected.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise ApprovalError(
            "approval receipt does not match the experiment: " + ", ".join(mismatches)
        )
    if not str(receipt.get("approved_by", "")).strip():
        raise ApprovalError("approval receipt has no approver")
    return receipt


def approval_path(experiment_id: str) -> Path:
    return resolve_editable_path(
        Path("experiments") / "approvals" / f"{experiment_id}.json"
    )


def _successful_record(
    registry: ExperimentRegistry, experiment_id: str
) -> dict[str, Any]:
    for record in registry.records():
        if record.get("experiment_id") == experiment_id:
            if record.get("status") != "success":
                raise ApprovalError(
                    f"experiment is not successful and cannot be approved: {experiment_id}"
                )
            return record
    raise ApprovalError(f"experiment is not registered: {experiment_id}")


def _require_campaign_finalist(record: dict[str, Any]) -> None:
    """Require the Phase 6 replicated +epsilon finalist bar before test access."""
    primary = float(record.get("metrics", {}).get("valid", {}).get("primary", -1.0))
    baseline = load_baseline_reference().primary
    if primary - baseline < 0.002:
        raise ApprovalError("campaign finalist must improve validation primary by at least 0.002")
    result_path = record.get("result_path")
    if not isinstance(result_path, str):
        raise ApprovalError("campaign finalist is missing result evidence")
    result = resolve_editable_path(result_path)
    try:
        value = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"could not read campaign result evidence: {exc}") from exc
    if len(value.get("member_metrics", [])) < 3:
        raise ApprovalError("campaign finalist requires three-seed result evidence")


def _exclusive_json_write(path: Path, value: Any) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ApprovalError(f"approval already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="use an isolated campaign workspace")
    parser.add_argument("experiment_id")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    try:
        if args.campaign:
            from experiment_engine.campaign import configure_campaign
            configure_campaign(args.campaign)
        receipt = grant_final_approval(
            args.experiment_id,
            approved_by=args.approved_by,
            confirmation=args.confirm,
        )
    except ApprovalError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
