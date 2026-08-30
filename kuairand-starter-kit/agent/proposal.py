"""Constrained Gemini proposal layer for governed pipeline experiments.

Gemini may choose one reviewed pipeline operator or one bounded scalar change.
The deterministic specification validator remains the authority for what may
execute, and training and state mutation remain in the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from agent.config import VertexConfig
from candidates.feature_pipeline import PIPELINE_STAGES, operator_contracts
from experiment_engine.experiment_spec import ExperimentSpec
from experiment_engine.experiment_templates import TEMPLATES, get_template


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
        for attempt in range(2):
            response = client.models.generate_content(
                model=self.config.model,
                contents=json.dumps(prompt, sort_keys=True),
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            try:
                return parse_proposal(response.text or "")
            except ValueError as exc:
                if attempt == 1:
                    raise
                prompt["correction"] = (
                    f"Previous proposal was invalid: {exc}. Return one JSON object "
                    "that satisfies every enumerated stage, operator, and one-change rule."
                )
        raise RuntimeError("Gemini did not return a valid proposal")
