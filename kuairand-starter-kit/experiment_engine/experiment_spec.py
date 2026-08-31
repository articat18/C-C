"""Strict JSON experiment specification used by the deterministic runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from candidates.feature_pipeline import validate_pipeline_selection
from experiment_engine.experiment_templates import get_template
from experiment_boundary import MAX_WALL_SECONDS, REPOSITORY_ROOT


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
EXPERIMENT_ID = re.compile(r"E[0-9]{4,8}")
V1_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_id",
    "template",
    "data_dir",
    "seed",
    "parameters",
    "budget",
    "hypothesis",
}
V2_TOP_LEVEL_FIELDS = V1_TOP_LEVEL_FIELDS | {
    "stage",
    "operator",
    "evidence",
    "expected_effect",
    "provenance",
}
MAX_EPOCHS = 40

PARAMETER_STAGES = {
    "bpr_weight": "loss",
    "bce_weight": "loss",
    "embedding_dim": "model",
    "popularity_weight": "model",
    "learning_rate": "training",
    "l2": "training",
    "patience": "training",
    "negatives_per_positive": "training",
}


class SpecificationError(ValueError):
    """Raised when an experiment specification is invalid or unsafe."""


@dataclass(frozen=True)
class ExperimentBudget:
    max_epochs: int = MAX_EPOCHS
    max_wall_seconds: int = MAX_WALL_SECONDS

    @classmethod
    def from_mapping(cls, value: Any) -> "ExperimentBudget":
        if not isinstance(value, Mapping):
            raise SpecificationError("budget must be a JSON object")
        unknown = sorted(set(value) - {"max_epochs", "max_wall_seconds"})
        if unknown:
            raise SpecificationError(f"unknown budget fields: {', '.join(unknown)}")
        max_epochs = value.get("max_epochs", MAX_EPOCHS)
        max_wall_seconds = value.get("max_wall_seconds", MAX_WALL_SECONDS)
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int):
            raise SpecificationError("budget.max_epochs must be an integer")
        if not 1 <= max_epochs <= MAX_EPOCHS:
            raise SpecificationError(
                f"budget.max_epochs must be between 1 and {MAX_EPOCHS}"
            )
        if isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, int):
            raise SpecificationError("budget.max_wall_seconds must be an integer")
        if not 1 <= max_wall_seconds <= MAX_WALL_SECONDS:
            raise SpecificationError(
                "budget.max_wall_seconds must be between 1 and "
                f"{MAX_WALL_SECONDS}"
            )
        return cls(max_epochs=max_epochs, max_wall_seconds=max_wall_seconds)


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    experiment_id: str
    template: str
    data_dir: str
    seed: int
    parameters: Mapping[str, int | float]
    budget: ExperimentBudget
    hypothesis: str
    stage: str = "training"
    operator: str = "none"
    evidence: str = ""
    expected_effect: str = ""
    provenance: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "ExperimentSpec":
        if not isinstance(value, Mapping):
            raise SpecificationError("experiment specification must be a JSON object")
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or schema_version not in SUPPORTED_SCHEMA_VERSIONS
        ):
            raise SpecificationError(
                f"schema_version must be one of {list(SUPPORTED_SCHEMA_VERSIONS)}; "
                f"received {schema_version!r}"
            )
        allowed_fields = (
            V1_TOP_LEVEL_FIELDS if schema_version == 1 else V2_TOP_LEVEL_FIELDS
        )
        unknown = sorted(set(value) - allowed_fields)
        if unknown:
            raise SpecificationError(
                f"unknown specification fields: {', '.join(unknown)}"
            )
        missing = sorted(
            {"schema_version", "experiment_id", "template", "hypothesis"}
            - set(value)
        )
        if missing:
            raise SpecificationError(f"missing required fields: {', '.join(missing)}")
        experiment_id = value["experiment_id"]
        if not isinstance(experiment_id, str) or not EXPERIMENT_ID.fullmatch(experiment_id):
            raise SpecificationError("experiment_id must match E followed by 4-8 digits")
        template_name = value["template"]
        if not isinstance(template_name, str):
            raise SpecificationError("template must be a string")
        template = get_template(template_name)
        raw_parameters = value.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise SpecificationError("parameters must be a JSON object")
        parameters = template.normalize_parameters(raw_parameters)
        seed = value.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise SpecificationError("seed must be an integer between 0 and 2^32-1")
        hypothesis = value["hypothesis"]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise SpecificationError("hypothesis must be a non-empty string")
        if len(hypothesis) > 2000:
            raise SpecificationError("hypothesis must contain at most 2000 characters")
        if schema_version == 1:
            stage = infer_pipeline_stage(template_name, raw_parameters)
            operator = "none"
            evidence = ""
            expected_effect = ""
        else:
            typed_missing = sorted(
                {"stage", "operator", "evidence", "expected_effect"} - set(value)
            )
            if typed_missing:
                raise SpecificationError(
                    f"schema version 2 is missing required fields: {', '.join(typed_missing)}"
                )
            stage = value["stage"]
            operator = value["operator"]
            evidence = value["evidence"]
            expected_effect = value["expected_effect"]
            if not isinstance(stage, str) or not isinstance(operator, str):
                raise SpecificationError("stage and operator must be strings")
            try:
                validate_pipeline_selection(stage, operator)
            except ValueError as exc:
                raise SpecificationError(str(exc)) from exc
            for field_name, field_value in (
                ("evidence", evidence),
                ("expected_effect", expected_effect),
            ):
                if not isinstance(field_value, str) or not field_value.strip():
                    raise SpecificationError(f"{field_name} must be a non-empty string")
                if len(field_value) > 4000:
                    raise SpecificationError(
                        f"{field_name} must contain at most 4000 characters"
                    )
            evidence = evidence.strip()
            expected_effect = expected_effect.strip()
            _validate_one_change(template_name, parameters, stage, operator)
        provenance = _validate_provenance(value.get("provenance"))
        data_dir = value.get("data_dir", "./KuaiRand-Pure/data")
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise SpecificationError("data_dir must be a non-empty string")
        _resolve_repository_path(data_dir)
        return cls(
            schema_version=schema_version,
            experiment_id=experiment_id,
            template=template_name,
            data_dir=data_dir,
            seed=seed,
            parameters=parameters,
            budget=ExperimentBudget.from_mapping(value.get("budget", {})),
            hypothesis=hypothesis.strip(),
            stage=stage,
            operator=operator,
            evidence=evidence,
            expected_effect=expected_effect,
            provenance=provenance,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        try:
            with Path(path).open(encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecificationError(f"could not load specification {path}: {exc}") from exc
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "template": self.template,
            "data_dir": self.data_dir,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "budget": {
                "max_epochs": self.budget.max_epochs,
                "max_wall_seconds": self.budget.max_wall_seconds,
            },
            "hypothesis": self.hypothesis,
        }
        if self.schema_version >= 2:
            value.update({
                "stage": self.stage,
                "operator": self.operator,
                "evidence": self.evidence,
                "expected_effect": self.expected_effect,
            })
            if self.provenance is not None:
                value["provenance"] = dict(self.provenance)
        return value

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def resolved_data_dir(self) -> Path:
        return _resolve_repository_path(self.data_dir)


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise SpecificationError(f"data_dir must stay inside the repository: {resolved}") from exc
    return resolved


def infer_pipeline_stage(
    template_name: str, parameters: Mapping[str, Any] | None
) -> str:
    """Infer the scalar-change stage for compatibility with older call sites."""

    template = get_template(template_name)
    normalized = template.normalize_parameters(parameters or {})
    defaults = template.normalize_parameters({})
    changed = [name for name, value in normalized.items() if value != defaults[name]]
    if len(changed) == 1:
        return PARAMETER_STAGES[changed[0]]
    return "training"


def _validate_one_change(
    template_name: str,
    parameters: Mapping[str, int | float],
    stage: str,
    operator: str,
) -> None:
    template = get_template(template_name)
    defaults = template.normalize_parameters({})
    changed = [name for name, value in parameters.items() if value != defaults[name]]
    if operator != "none":
        if changed:
            raise SpecificationError(
                "cleaning and feature experiments must hold model parameters at template defaults"
            )
        return
    if len(changed) > 1:
        raise SpecificationError(
            f"schema version 2 permits one scalar change; received {sorted(changed)}"
        )
    if changed and PARAMETER_STAGES[changed[0]] != stage:
        raise SpecificationError(
            f"parameter {changed[0]!r} belongs to stage "
            f"{PARAMETER_STAGES[changed[0]]!r}, not {stage!r}"
        )


def _validate_provenance(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpecificationError("provenance must be a JSON object")
    allowed = {
        "proposal_fingerprint", "context_fingerprint", "source_model",
        "token_usage", "manual_interventions",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecificationError(
            f"unknown provenance fields: {', '.join(unknown)}"
        )
    proposal_hash = value.get("proposal_fingerprint")
    if not isinstance(proposal_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", proposal_hash
    ):
        raise SpecificationError("provenance.proposal_fingerprint must be SHA-256")
    context_hash = value.get("context_fingerprint")
    if context_hash is not None and (
        not isinstance(context_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", context_hash)
    ):
        raise SpecificationError(
            "provenance.context_fingerprint must be SHA-256 or null"
        )
    source_model = value.get("source_model")
    if source_model is not None and (
        not isinstance(source_model, str) or not source_model.strip()
    ):
        raise SpecificationError("provenance.source_model must be a string or null")
    usage = value.get("token_usage", {})
    if not isinstance(usage, Mapping):
        raise SpecificationError("provenance.token_usage must be an object")
    normalized_usage: dict[str, int] = {}
    for name, amount in usage.items():
        if (
            not isinstance(name, str)
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
        ):
            raise SpecificationError(
                "provenance.token_usage values must be non-negative integers"
            )
        normalized_usage[name] = amount
    interventions = value.get("manual_interventions", 0)
    if (
        isinstance(interventions, bool)
        or not isinstance(interventions, int)
        or interventions < 0
    ):
        raise SpecificationError(
            "provenance.manual_interventions must be a non-negative integer"
        )
    return {
        "proposal_fingerprint": proposal_hash,
        "context_fingerprint": context_hash,
        "source_model": source_model,
        "token_usage": normalized_usage,
        "manual_interventions": interventions,
    }
