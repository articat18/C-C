"""Validated environment configuration for Vertex AI access."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required Vertex AI configuration is absent or unsafe."""


@dataclass(frozen=True)
class VertexConfig:
    project: str
    location: str
    model: str
    use_vertex_ai: bool = True

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "VertexConfig":
        if environment is None:
            load_dotenv()
            env = os.environ
        else:
            env = environment
        raw_vertex_flag = env.get("GOOGLE_GENAI_USE_VERTEXAI", "true").strip().lower()
        if raw_vertex_flag not in {"1", "true", "yes"}:
            raise ConfigurationError(
                "GOOGLE_GENAI_USE_VERTEXAI must be true for this agent"
            )

        config = cls(
            project=env.get("GOOGLE_CLOUD_PROJECT", "").strip(),
            location=env.get("GOOGLE_CLOUD_LOCATION", "").strip(),
            model=env.get("VERTEX_MODEL", "").strip(),
        )
        missing = [
            name
            for name, value in (
                ("GOOGLE_CLOUD_PROJECT", config.project),
                ("GOOGLE_CLOUD_LOCATION", config.location),
                ("VERTEX_MODEL", config.model),
            )
            if not value or value.startswith(("your-", "replace-"))
        ]
        if missing:
            raise ConfigurationError(
                "missing Vertex AI configuration: " + ", ".join(missing)
            )
        return config
