"""Constrained Gemini proposal layer for governed pipeline experiments.

Gemini may choose one reviewed pipeline operator or one bounded scalar change.
The deterministic specification validator remains the authority for what may
execute, and training and state mutation remain in the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from agent.config import VertexConfig
from candidates.feature_pipeline import PIPELINE_STAGES, operator_contracts
from experiment_engine.checkpoints import _atomic_json_write
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.experiment_templates import TEMPLATES, get_template
from experiment_boundary import resolve_editable_path


PROPOSAL_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ExperimentProposal:
    template: str
    stage: str
    operator: str
    hypothesis: str
    evidence: str
    expected_effect: str
    parameters: dict[str, int | float]
    seed: int = 0
    provenance: dict[str, Any] = field(default_factory=dict, compare=False)


def proposal_to_dict(proposal: ExperimentProposal) -> dict[str, Any]:
    """Return the executable proposal fields, excluding transport metadata."""

    return {
        "template": proposal.template,
        "stage": proposal.stage,
        "operator": proposal.operator,
        "hypothesis": proposal.hypothesis,
        "evidence": proposal.evidence,
        "expected_effect": proposal.expected_effect,
        "parameters": dict(proposal.parameters),
        "seed": proposal.seed,
    }


def proposal_fingerprint(proposal: ExperimentProposal) -> str:
    encoded = json.dumps(
        proposal_to_dict(proposal), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_fingerprint(context: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(context), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_proposal_artifact(
    path: str | Path,
    proposal: ExperimentProposal,
) -> Path:
    """Atomically save one validated proposal and its integrity metadata."""

    destination = resolve_editable_path(path)
    fingerprint = proposal_fingerprint(proposal)
    provenance = dict(proposal.provenance)
    artifact = {
        "artifact_version": PROPOSAL_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal": proposal_to_dict(proposal),
        "proposal_fingerprint": fingerprint,
        "context_fingerprint": provenance.get("context_fingerprint"),
        "source_model": provenance.get("source_model"),
        "token_usage": provenance.get("token_usage", {}),
    }
    if destination.exists():
        raise ValueError(f"proposal artifact already exists: {destination}")
    _atomic_json_write(destination, artifact)
    return destination


def load_proposal_artifact(path: str | Path) -> ExperimentProposal:
    """Load, revalidate, and integrity-check a saved proposal artifact."""

    source = resolve_editable_path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load proposal artifact {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("proposal artifact must be a JSON object")
    allowed = {
        "artifact_version", "created_at", "proposal", "proposal_fingerprint",
        "context_fingerprint", "source_model", "token_usage",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"proposal artifact contains unsupported fields: {', '.join(unknown)}"
        )
    if value.get("artifact_version") != PROPOSAL_ARTIFACT_VERSION:
        raise ValueError(
            f"proposal artifact version must be {PROPOSAL_ARTIFACT_VERSION}"
        )
    proposal = parse_proposal(value.get("proposal"))
    expected = value.get("proposal_fingerprint")
    observed = proposal_fingerprint(proposal)
    if not isinstance(expected, str) or expected != observed:
        raise ValueError("proposal artifact fingerprint mismatch; file was modified")
    provenance = _validate_provenance({
        "proposal_fingerprint": observed,
        "context_fingerprint": value.get("context_fingerprint"),
        "source_model": value.get("source_model"),
        "token_usage": value.get("token_usage", {}),
    })
    return replace(proposal, provenance=provenance)


def parse_proposal(value: str | Mapping[str, Any]) -> ExperimentProposal:
    """Validate a model proposal without permitting code or unknown fields."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"proposal is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("proposal must be a JSON object")
    allowed = {
        "template", "stage", "operator", "hypothesis", "evidence",
        "expected_effect", "parameters", "seed",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"proposal contains unsupported fields: {', '.join(unknown)}")
    template = value.get("template")
    stage = value.get("stage")
    operator = value.get("operator")
    hypothesis = value.get("hypothesis")
    evidence = value.get("evidence")
    expected_effect = value.get("expected_effect")
    required_strings = {
        "template": template,
        "stage": stage,
        "operator": operator,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "expected_effect": expected_effect,
    }
    missing = [
        name for name, field in required_strings.items()
        if not isinstance(field, str) or not field.strip()
    ]
    if missing:
        raise ValueError(
            f"proposal requires non-empty string fields: {', '.join(sorted(missing))}"
        )
    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("proposal.parameters must be an object")
    normalized = get_template(template).normalize_parameters(parameters)
    seed = value.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("proposal.seed must be an integer between 0 and 2^32-1")
    try:
        validated = ExperimentSpec.from_mapping({
            "schema_version": 2,
            "experiment_id": "E0000",
            "template": template,
            "stage": stage,
            "operator": operator,
            "hypothesis": hypothesis,
            "evidence": evidence,
            "expected_effect": expected_effect,
            "parameters": dict(normalized),
            "seed": seed,
        })
    except ValueError as exc:
        raise ValueError(f"proposal violates the experiment contract: {exc}") from exc
    return ExperimentProposal(
        template=validated.template,
        stage=validated.stage,
        operator=validated.operator,
        hypothesis=validated.hypothesis,
        evidence=validated.evidence,
        expected_effect=validated.expected_effect,
        parameters=dict(validated.parameters),
        seed=validated.seed,
    )


class GeminiProposalClient:
    def __init__(self, config: VertexConfig | None = None) -> None:
        self.config = config or VertexConfig.from_environment()

    def propose(self, context: Mapping[str, Any]) -> ExperimentProposal:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed") from exc
        client = genai.Client(
            vertexai=True, project=self.config.project, location=self.config.location
        )
        allowed_parameters = {
            name: sorted(get_template(name).parameters)
            for name in sorted(TEMPLATES)
        }
        prompt = {
            "task": "Propose one evidence-backed Phase 4 experiment for the governed AutoML engine.",
            "allowed_stages": list(PIPELINE_STAGES),
            "allowed_templates": allowed_parameters,
            "allowed_operators": operator_contracts(),
            "strict_rules": [
                "Change exactly one reviewed operator or one scalar parameter from its template default.",
                "Cleaning and feature operators must keep every model parameter at its template default.",
                "Use operator 'none' for loss, model, or training changes.",
                "Do not propose source code, commands, file paths, hidden-test access, or unlisted operators.",
            ],
            "required_output": {
                "template": "string",
                "stage": "string",
                "operator": "string",
                "hypothesis": "string",
                "evidence": "string",
                "expected_effect": "string",
                "parameters": "object",
                "seed": "integer",
            },
            "context": dict(context),
        }
        accumulated_usage: dict[str, int] = {}
        for attempt in range(2):
            response = client.models.generate_content(
                model=self.config.model,
                contents=json.dumps(prompt, sort_keys=True),
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            for name, amount in _extract_token_usage(response).items():
                accumulated_usage[name] = accumulated_usage.get(name, 0) + amount
            try:
                proposal = parse_proposal(response.text or "")
                return replace(proposal, provenance=_validate_provenance({
                    "proposal_fingerprint": proposal_fingerprint(proposal),
                    "context_fingerprint": context_fingerprint(context),
                    "source_model": self.config.model,
                    "token_usage": accumulated_usage,
                }))
            except ValueError as exc:
                if attempt == 1:
                    raise
                prompt["correction"] = (
                    f"Previous proposal was invalid: {exc}. Return one JSON object "
                    "that satisfies every enumerated stage, operator, and one-change rule."
                )
        raise RuntimeError("Gemini did not return a valid proposal")


def _extract_token_usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    names = {
        "input_tokens": ("prompt_token_count", "input_token_count"),
        "output_tokens": ("candidates_token_count", "output_token_count"),
        "total_tokens": ("total_token_count",),
        "cached_tokens": ("cached_content_token_count",),
        "thought_tokens": ("thoughts_token_count",),
    }
    usage: dict[str, int] = {}
    for output_name, candidates in names.items():
        for candidate in candidates:
            value = getattr(metadata, candidate, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[output_name] = value
                break
    return usage


def _validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint = value.get("proposal_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("proposal provenance requires a SHA-256 fingerprint")
    context_hash = value.get("context_fingerprint")
    if context_hash is not None and (
        not isinstance(context_hash, str) or len(context_hash) != 64
    ):
        raise ValueError("context fingerprint must be a SHA-256 string or null")
    source_model = value.get("source_model")
    if source_model is not None and (
        not isinstance(source_model, str) or not source_model.strip()
    ):
        raise ValueError("source model must be a non-empty string or null")
    token_usage = value.get("token_usage", {})
    if not isinstance(token_usage, Mapping):
        raise ValueError("token usage must be an object")
    normalized_usage: dict[str, int] = {}
    for name, amount in token_usage.items():
        if (
            not isinstance(name, str)
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
        ):
            raise ValueError("token usage values must be non-negative integers")
        normalized_usage[name] = amount
    return {
        "proposal_fingerprint": fingerprint,
        "context_fingerprint": context_hash,
        "source_model": source_model,
        "token_usage": normalized_usage,
    }
