"""Constrained Gemini proposal layer for the deterministic orchestrator.

Gemini may suggest an experiment, but this module accepts only JSON that matches
the closed experiment-template catalogue.  Training and state mutation remain in
the deterministic controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from agent.config import VertexConfig
from experiment_engine.experiment_templates import get_template


@dataclass(frozen=True)
class ExperimentProposal:
    template: str
    hypothesis: str
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
    allowed = {"template", "hypothesis", "parameters", "seed"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"proposal contains unsupported fields: {', '.join(unknown)}")
    template = value.get("template")
    hypothesis = value.get("hypothesis")
    if not isinstance(template, str) or not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("proposal requires a template and non-empty hypothesis")
    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("proposal.parameters must be an object")
    normalized = get_template(template).normalize_parameters(parameters)
    seed = value.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("proposal.seed must be an integer between 0 and 2^32-1")
    return ExperimentProposal(template, hypothesis.strip(), dict(normalized), seed)


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
        prompt = {
            "task": "Propose one bounded experiment for the deterministic AutoML engine.",
            "allowed_templates": ["bpr_hybrid", "bpr_ensemble"],
            "required_output": {"template": "string", "hypothesis": "string", "parameters": "object", "seed": "integer"},
            "context": dict(context),
        }
        response = client.models.generate_content(
            model=self.config.model,
            contents=json.dumps(prompt, sort_keys=True),
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return parse_proposal(response.text or "")
