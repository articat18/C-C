"""Validate local Vertex configuration and optionally make one live API call."""

from __future__ import annotations

import argparse

from agent.config import ConfigurationError, VertexConfig


def run_live_check(config: VertexConfig) -> None:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed; install requirements-agent.txt"
        ) from exc

    client = genai.Client(
        vertexai=True,
        project=config.project,
        location=config.location,
    )
    response = client.models.generate_content(
        model=config.model,
        contents="Return exactly VERTEX_OK and nothing else.",
    )
    if (response.text or "").strip() != "VERTEX_OK":
        raise RuntimeError(
            f"Vertex AI returned an unexpected health-check response: {response.text!r}"
        )
    print("live Vertex AI health check passed: VERTEX_OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="make one potentially billable Vertex AI model request",
    )
    args = parser.parse_args()
    try:
        config = VertexConfig.from_environment()
    except ConfigurationError as exc:
        parser.error(str(exc))
    print(
        "Vertex configuration valid: "
        f"project={config.project} location={config.location} model={config.model}"
    )
    if args.live:
        run_live_check(config)
    else:
        print("configuration-only check passed; no API request was made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
