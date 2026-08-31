"""Validate bounded candidate-code patches in a disposable repository copy."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping

from agent.config import VertexConfig
from agent.proposal import _extract_token_usage, context_fingerprint
from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_templates import get_template
from experiment_engine.campaign import active_campaign
from experiment_boundary import REPOSITORY_ROOT, resolve_editable_path


PATCH_ARTIFACT_VERSION = 1
PROMOTION_CONFIRMATION = "PROMOTE REVIEWED CANDIDATE PATCH"
MAX_PATCH_BYTES = 64 * 1024
MAX_PATCH_FILES = 8
ALLOWED_CONTRACTS = {"feature_operator", "cleaning_operator", "training_strategy"}
BANNED_IMPORT_ROOTS = {
    "ctypes", "http", "importlib", "multiprocessing", "os", "pickle",
    "pathlib", "requests", "shutil", "socket", "subprocess", "tempfile", "urllib",
}
BANNED_CALLS = {"compile", "eval", "exec", "input", "open", "__import__"}
BANNED_ATTRIBUTES = {"popen", "remove", "rmdir", "rmtree", "system", "unlink"}


class CandidatePatchError(ValueError):
    """Raised when a candidate patch crosses its bounded contract."""


@dataclass(frozen=True)
class CandidatePatchProposal:
    rationale: str
    affected_contract: str
    patch: str
    experiment: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.patch.encode("utf-8")).hexdigest()


def parse_candidate_patch(value: Mapping[str, Any]) -> CandidatePatchProposal:
    if not isinstance(value, Mapping):
        raise CandidatePatchError("candidate patch proposal must be a JSON object")
    unknown = sorted(
        set(value) - {"rationale", "affected_contract", "patch", "experiment"}
    )
    if unknown:
        raise CandidatePatchError(f"unsupported patch fields: {', '.join(unknown)}")
    rationale = value.get("rationale")
    contract = value.get("affected_contract")
    patch = value.get("patch")
    if not isinstance(rationale, str) or not rationale.strip():
        raise CandidatePatchError("patch rationale must be a non-empty string")
    if contract not in ALLOWED_CONTRACTS:
        raise CandidatePatchError(
            f"affected_contract must be one of {sorted(ALLOWED_CONTRACTS)}"
        )
    if not isinstance(patch, str) or not patch.strip():
        raise CandidatePatchError("patch must be a non-empty unified diff")
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise CandidatePatchError(f"patch exceeds {MAX_PATCH_BYTES} bytes")
    _touched_paths(patch)
    experiment = _validate_patch_experiment(value.get("experiment"))
    return CandidatePatchProposal(rationale.strip(), contract, patch, experiment)


def save_candidate_patch_artifact(
    path: str | Path,
    proposal: CandidatePatchProposal,
) -> Path:
    destination = resolve_editable_path(path)
    if destination.exists():
        raise CandidatePatchError(f"patch artifact already exists: {destination}")
    _atomic_json_write(destination, {
        "artifact_version": PATCH_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal": {
            "rationale": proposal.rationale,
            "affected_contract": proposal.affected_contract,
            "patch": proposal.patch,
            "experiment": proposal.experiment,
        },
        "content_hash": proposal.content_hash,
        "context_fingerprint": proposal.provenance.get("context_fingerprint"),
        "source_model": proposal.provenance.get("source_model"),
        "token_usage": proposal.provenance.get("token_usage", {}),
    })
    return destination


def load_candidate_patch_artifact(path: str | Path) -> CandidatePatchProposal:
    source = resolve_editable_path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidatePatchError(f"could not load patch artifact: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CandidatePatchError("patch artifact must be a JSON object")
    if set(value) != {
        "artifact_version", "created_at", "proposal", "content_hash",
        "context_fingerprint", "source_model", "token_usage",
    }:
        raise CandidatePatchError("patch artifact fields do not match version 1")
    if value.get("artifact_version") != PATCH_ARTIFACT_VERSION:
        raise CandidatePatchError("unsupported patch artifact version")
    proposal = parse_candidate_patch(value.get("proposal"))
    if value.get("content_hash") != proposal.content_hash:
        raise CandidatePatchError("patch artifact fingerprint mismatch")
    return replace(proposal, provenance={
        "context_fingerprint": value.get("context_fingerprint"),
        "source_model": value.get("source_model"),
        "token_usage": value.get("token_usage", {}),
    })


class GeminiCandidatePatchClient:
    """Generate one bounded candidates-only patch without applying it."""

    def __init__(self, config: VertexConfig | None = None) -> None:
        self.config = config or VertexConfig.from_environment()

    def propose(self, context: Mapping[str, Any]) -> CandidatePatchProposal:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed") from exc
        client = genai.Client(
            vertexai=True, project=self.config.project, location=self.config.location
        )
        prompt: dict[str, Any] = {
            "task": "Propose one bounded Phase 4 candidate-code patch.",
            "required_output": {
                "rationale": "string",
                "affected_contract": sorted(ALLOWED_CONTRACTS),
                "patch": "unified diff string",
                "experiment": {
                    "template": "bpr_hybrid or bpr_ensemble",
                    "stage": "cleaning, features, loss, model, or training",
                    "operator": "operator introduced by the patch",
                    "hypothesis": "string",
                    "evidence": "string",
                    "expected_effect": "string",
                    "parameters": "object containing template defaults",
                    "seed": "integer",
                },
            },
            "strict_rules": [
                "Touch at most eight Python files, all under candidates/.",
                "Do not add commands, arbitrary file access, network access, subprocesses, dynamic imports, or serialization loaders.",
                "Preserve row order and count and fit learned state on training only.",
                "Do not access validation or test labels while fitting features.",
                "Return JSON only and include the complete unified diff in patch.",
            ],
            "context": dict(context),
        }
        usage: dict[str, int] = {}
        for attempt in range(2):
            response = client.models.generate_content(
                model=self.config.model,
                contents=json.dumps(prompt, sort_keys=True),
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            for name, amount in _extract_token_usage(response).items():
                usage[name] = usage.get(name, 0) + amount
            try:
                raw = json.loads(response.text or "")
                proposal = parse_candidate_patch(raw)
                return replace(proposal, provenance={
                    "context_fingerprint": context_fingerprint(context),
                    "source_model": self.config.model,
                    "token_usage": usage,
                })
            except (json.JSONDecodeError, CandidatePatchError) as exc:
                if attempt == 1:
                    raise CandidatePatchError(f"invalid model patch proposal: {exc}") from exc
                prompt["correction"] = f"Previous patch proposal was invalid: {exc}"
        raise RuntimeError("Gemini did not return a valid candidate patch")


def validate_candidate_patch(proposal: CandidatePatchProposal) -> dict[str, Any]:
    """Apply and verify a patch in a clean disposable checkout."""

    touched = _touched_paths(proposal.patch)
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="candidate-patch-") as directory:
        sandbox = Path(directory)
        project = _extract_clean_checkout(sandbox)
        _run(
            ["git", "apply", "--check", "-"],
            cwd=project,
            input_text=proposal.patch,
            name="patch_apply_check",
            checks=checks,
        )
        _run(
            ["git", "apply", "-"],
            cwd=project,
            input_text=proposal.patch,
            name="patch_apply",
            checks=checks,
        )
        _static_policy_check(project, touched)
        checks.append({"name": "static_policy", "status": "passed"})
        commands = (
            ("syntax", [sys.executable, "-m", "compileall", "-q", "candidates"]),
            ("imports", [sys.executable, "-c", "import candidates.feature_pipeline"]),
            ("protected_boundary", [sys.executable, "experiment_boundary.py", "--check"]),
            ("leakage_alignment", [
                sys.executable, "-m", "unittest", "tests.test_feature_pipeline",
                "tests.test_history_features", "tests.test_experiment_runner",
            ]),
            ("full_test_suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests"]),
        )
        for name, command in commands:
            _run(command, cwd=project, name=name, checks=checks, timeout=600)
        experiment_result = None
        if proposal.experiment is not None:
            experiment_result = _run_sandbox_experiment(
                project, proposal.experiment, checks
            )
    report = {
        "status": "accepted",
        "content_hash": proposal.content_hash,
        "affected_contract": proposal.affected_contract,
        "touched_paths": [path.as_posix() for path in touched],
        "checks": checks,
        "duration_seconds": round(time.monotonic() - started, 6),
        "promoted": False,
    }
    if experiment_result is not None:
        report["experiment_result"] = experiment_result
    return report


def promote_candidate_patch(
    proposal: CandidatePatchProposal,
    report: Mapping[str, Any],
    *,
    confirmation: str,
) -> dict[str, Any]:
    """Atomically apply a previously accepted patch after explicit review."""

    if confirmation != PROMOTION_CONFIRMATION:
        raise CandidatePatchError("candidate patch promotion requires explicit confirmation")
    return _promote_candidate_patch(proposal, report, manual_interventions=1)


def auto_promote_candidate_patch(
    proposal: CandidatePatchProposal,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote a Phase 6 patch only after complete sandbox and result evidence."""
    if active_campaign() is None:
        raise CandidatePatchError("automatic promotion is available only inside a campaign")
    return _promote_candidate_patch(proposal, report, manual_interventions=0)


def _promote_candidate_patch(
    proposal: CandidatePatchProposal,
    report: Mapping[str, Any],
    *,
    manual_interventions: int,
) -> dict[str, Any]:
    if report.get("status") != "accepted":
        raise CandidatePatchError("candidate patch report is not accepted")
    if report.get("content_hash") != proposal.content_hash:
        raise CandidatePatchError("candidate patch report fingerprint mismatch")
    expected_paths = [path.as_posix() for path in _touched_paths(proposal.patch)]
    if report.get("touched_paths") != expected_paths:
        raise CandidatePatchError("candidate patch report paths do not match proposal")
    experiment_result = report.get("experiment_result", {})
    if experiment_result.get("comparison", {}).get("decision") != "keep":
        raise CandidatePatchError(
            "candidate patch promotion requires successful sandbox result evidence"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "candidates"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise CandidatePatchError("candidates/ must be clean before patch promotion")
    subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=REPOSITORY_ROOT,
        input=proposal.patch,
        text=True,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "apply", "-"],
        cwd=REPOSITORY_ROOT,
        input=proposal.patch,
        text=True,
        check=True,
        capture_output=True,
    )
    return {
        "status": "promoted",
        "content_hash": proposal.content_hash,
        "touched_paths": expected_paths,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "manual_interventions": manual_interventions,
        "promotion_policy": "autonomous_verified" if not manual_interventions else "manual_reviewed",
    }


def _touched_paths(patch: str) -> tuple[PurePosixPath, ...]:
    paths: set[PurePosixPath] = set()
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        if raw.startswith(("a/", "b/")):
            raw = raw[2:]
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            raise CandidatePatchError(f"unsafe patch path: {raw}")
        if len(path.parts) < 2 or path.parts[0] != "candidates":
            raise CandidatePatchError(f"patch path is outside candidates/: {raw}")
        if path.suffix != ".py":
            raise CandidatePatchError(f"candidate patch may change only Python: {raw}")
        paths.add(path)
    if not paths:
        raise CandidatePatchError("patch contains no candidates/*.py paths")
    if len(paths) > MAX_PATCH_FILES:
        raise CandidatePatchError(f"patch changes more than {MAX_PATCH_FILES} files")
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _validate_patch_experiment(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CandidatePatchError("patch experiment must be a JSON object")
    allowed = {
        "template", "stage", "operator", "hypothesis", "evidence",
        "expected_effect", "parameters", "seed", "budget",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CandidatePatchError(
            f"unsupported patch experiment fields: {', '.join(unknown)}"
        )
    strings = {
        name: value.get(name)
        for name in (
            "template", "stage", "operator", "hypothesis", "evidence",
            "expected_effect",
        )
    }
    if any(not isinstance(item, str) or not item.strip() for item in strings.values()):
        raise CandidatePatchError("patch experiment requires all descriptive fields")
    if strings["stage"] not in {"cleaning", "features", "loss", "model", "training"}:
        raise CandidatePatchError("unsupported patch experiment stage")
    template = get_template(strings["template"])
    parameters = template.normalize_parameters(value.get("parameters", {}))
    if parameters != template.normalize_parameters({}):
        raise CandidatePatchError("candidate patch experiment must use template defaults")
    seed = value.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise CandidatePatchError("patch experiment seed is invalid")
    budget = value.get("budget", {"max_epochs": 40, "max_wall_seconds": 600})
    if not isinstance(budget, Mapping) or set(budget) - {"max_epochs", "max_wall_seconds"}:
        raise CandidatePatchError("patch experiment budget is invalid")
    max_epochs = budget.get("max_epochs", 40)
    max_wall_seconds = budget.get("max_wall_seconds", 600)
    if not isinstance(max_epochs, int) or not 1 <= max_epochs <= 40:
        raise CandidatePatchError("patch experiment max_epochs must be 1..40")
    if not isinstance(max_wall_seconds, int) or not 1 <= max_wall_seconds <= 600:
        raise CandidatePatchError("patch experiment max_wall_seconds must be 1..600")
    return {
        **{name: item.strip() for name, item in strings.items()},
        "parameters": dict(parameters),
        "seed": seed,
        "budget": {
            "max_epochs": max_epochs,
            "max_wall_seconds": max_wall_seconds,
        },
    }


def _extract_clean_checkout(destination: Path) -> Path:
    git_root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    git_root = Path(git_root_result.stdout.strip()).resolve()
    relative_project = REPOSITORY_ROOT.relative_to(git_root)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=git_root,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(destination, filter="data")
    project = destination / relative_project
    if not project.is_dir():
        raise CandidatePatchError("clean checkout did not contain the project")
    return project


def _static_policy_check(project: Path, touched: tuple[PurePosixPath, ...]) -> None:
    for relative in touched:
        path = project / relative
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, SyntaxError) as exc:
            raise CandidatePatchError(f"invalid candidate Python {relative}: {exc}") from exc
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if not isinstance(
                node,
                (
                    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                    ast.ClassDef, ast.Assign, ast.AnnAssign,
                ),
            ):
                raise CandidatePatchError(
                    f"executable top-level statement in {relative}: {type(node).__name__}"
                )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if name.split(".", 1)[0] in BANNED_IMPORT_ROOTS:
                        raise CandidatePatchError(f"banned import in {relative}: {name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALLS:
                    raise CandidatePatchError(
                        f"banned call in {relative}: {node.func.id}"
                    )
            if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
                raise CandidatePatchError(
                    f"banned attribute in {relative}: {node.attr}"
                )


def _run_sandbox_experiment(
    project: Path,
    experiment: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    source_data = REPOSITORY_ROOT / "KuaiRand-Pure" / "data"
    sandbox_data = project / "KuaiRand-Pure" / "data"
    sandbox_data.parent.mkdir(parents=True, exist_ok=True)
    if not sandbox_data.exists():
        os.symlink(source_data, sandbox_data, target_is_directory=True)
    spec = {
        "schema_version": 2,
        "experiment_id": "E99999999",
        **experiment,
    }
    script = (
        "import json,sys; "
        "from experiment_engine.experiment_spec import ExperimentSpec; "
        "from experiment_engine.experiment_runner import run_experiment; "
        "spec=ExperimentSpec.from_mapping(json.load(sys.stdin)); "
        "print(json.dumps(run_experiment(spec, verbose=False), default=str))"
    )
    completed = _run(
        [sys.executable, "-c", script],
        cwd=project,
        input_text=json.dumps(spec),
        name="sandbox_training",
        checks=checks,
        timeout=int(experiment["budget"]["max_wall_seconds"]) + 30,
    )
    result = json.loads(completed.stdout)
    baseline = 0.6016
    primary = float(result["metrics"]["valid"]["primary"])
    result["comparison"] = {
        "reference": "stable_published_baseline",
        "previous_best": baseline,
        "candidate": primary,
        "improvement": primary - baseline,
        "epsilon": 0.002,
        "decision": "keep" if primary - baseline > 0.002 else "reject_or_refine",
    }
    return result


def _run(
    command: list[str],
    *,
    cwd: Path,
    name: str,
    checks: list[dict[str, Any]],
    input_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    check = {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "duration_seconds": round(time.monotonic() - started, 6),
        "output_tail": (completed.stdout + completed.stderr)[-2000:],
    }
    checks.append(check)
    if completed.returncode != 0:
        raise CandidatePatchError(f"{name} failed: {check['output_tail']}")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="use an isolated campaign workspace")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--promote-report", type=Path)
    parser.add_argument("--confirmation")
    parser.add_argument("--auto-promote", action="store_true")
    args = parser.parse_args()
    if args.campaign:
        from experiment_engine.campaign import configure_campaign
        configure_campaign(args.campaign)
    proposal = load_candidate_patch_artifact(args.artifact)
    if args.promote_report:
        report_path = resolve_editable_path(args.promote_report)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output = (
            auto_promote_candidate_patch(proposal, report)
            if args.auto_promote
            else promote_candidate_patch(proposal, report, confirmation=args.confirmation or "")
        )
    else:
        output = validate_candidate_patch(proposal)
    if args.report:
        destination = resolve_editable_path(args.report)
        if destination.exists():
            parser.error(f"report already exists: {destination}")
        _atomic_json_write(destination, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
