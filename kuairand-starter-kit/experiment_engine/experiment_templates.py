"""Closed catalog of experiment templates available to the orchestrator.

Specifications select one of these templates and provide scalar parameters.  They
cannot name Python objects, commands, or arbitrary source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


class TemplateValidationError(ValueError):
    """Raised when a template or one of its parameters is unsupported."""


@dataclass(frozen=True)
class ParameterRule:
    kind: type
    default: int | float
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[int | float, ...] | None = None

    def validate(self, name: str, value: Any) -> int | float:
        if isinstance(value, bool):
            raise TemplateValidationError(f"parameter {name!r} must be numeric")
        if self.kind is int:
            if not isinstance(value, int):
                raise TemplateValidationError(f"parameter {name!r} must be an integer")
            normalized: int | float = value
        elif self.kind is float:
            if not isinstance(value, Real):
                raise TemplateValidationError(f"parameter {name!r} must be numeric")
            normalized = float(value)
        else:  # Defensive: the catalog below only uses int and float.
            raise TypeError(f"unsupported rule kind: {self.kind}")
        if self.minimum is not None and normalized < self.minimum:
            raise TemplateValidationError(
                f"parameter {name!r} must be >= {self.minimum}"
            )
        if self.maximum is not None and normalized > self.maximum:
            raise TemplateValidationError(
                f"parameter {name!r} must be <= {self.maximum}"
            )
        if self.choices is not None and normalized not in self.choices:
            raise TemplateValidationError(
                f"parameter {name!r} must be one of {list(self.choices)}"
            )
        return normalized


@dataclass(frozen=True)
class ExperimentTemplate:
    name: str
    description: str
    parameters: Mapping[str, ParameterRule]
    ensemble_members: int = 1
    objective: str = "bpr_hybrid"

    def normalize_parameters(self, supplied: Mapping[str, Any]) -> dict[str, int | float]:
        unknown = sorted(set(supplied) - set(self.parameters))
        if unknown:
            raise TemplateValidationError(
                f"unsupported parameters for {self.name!r}: {', '.join(unknown)}"
            )
        return {
            name: rule.validate(name, supplied.get(name, rule.default))
            for name, rule in self.parameters.items()
        }


_BPR_PARAMETERS = {
    "embedding_dim": ParameterRule(int, 16, minimum=1, maximum=128),
    "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
    "l2": ParameterRule(float, 1e-5, minimum=0.0, maximum=0.1),
    "patience": ParameterRule(int, 4, minimum=1, maximum=20),
    "negatives_per_positive": ParameterRule(
        int, 4, choices=(1, 2, 4, 8)
    ),
    "bpr_weight": ParameterRule(float, 1.0, minimum=0.0, maximum=2.0),
    "bce_weight": ParameterRule(float, 0.1, minimum=0.0, maximum=2.0),
    "popularity_weight": ParameterRule(float, 0.0, minimum=0.0, maximum=0.5),
}


TEMPLATES: dict[str, ExperimentTemplate] = {
    "pointwise_fm": ExperimentTemplate(
        name="pointwise_fm",
        description=(
            "One pointwise BCE FM matching the protected official baseline and "
            "supporting reviewed candidate operators."
        ),
        parameters={
            "embedding_dim": ParameterRule(int, 16, minimum=1, maximum=128),
            "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
            "l2": ParameterRule(float, 1e-6, minimum=0.0, maximum=0.1),
            "patience": ParameterRule(int, 4, minimum=1, maximum=20),
            "batch_size": ParameterRule(int, 8192, choices=(2048, 4096, 8192, 16384)),
        },
        objective="pointwise_bce",
    ),
    "lambdarank_fm": ExperimentTemplate(
        name="lambdarank_fm",
        description=(
            "Warm-start a matched pointwise FM checkpoint and fine-tune it with "
            "delta-nDCG@5-weighted pairwise gradients at one tenth of the base "
            "learning rate."
        ),
        parameters={
            "embedding_dim": ParameterRule(int, 16, minimum=1, maximum=128),
            "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
            "l2": ParameterRule(float, 1e-6, minimum=0.0, maximum=0.1),
            "patience": ParameterRule(int, 4, minimum=1, maximum=20),
            "batch_size": ParameterRule(int, 8192, choices=(2048, 4096, 8192, 16384)),
        },
        objective="lambdarank",
    ),
    "sequence_mlp": ExperimentTemplate(
        name="sequence_mlp",
        description=(
            "Causal user-history embedding MLP. Its prior-positive-video context "
            "is fit from earlier training dates only."
        ),
        parameters={
            "embedding_dim": ParameterRule(int, 16, minimum=4, maximum=64),
            "hidden_dim": ParameterRule(int, 32, choices=(16, 32, 64)),
            "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
            "l2": ParameterRule(float, 1e-6, minimum=0.0, maximum=0.1),
            "patience": ParameterRule(int, 4, minimum=1, maximum=20),
            "batch_size": ParameterRule(int, 8192, choices=(2048, 4096, 8192, 16384)),
        },
        objective="sequence_mlp",
    ),
    "sequence_ensemble": ExperimentTemplate(
        name="sequence_ensemble",
        description=(
            "Three consecutive-seed causal sequence MLPs averaged before "
            "validation scoring; used only after matched single-seed replication."
        ),
        parameters={
            "embedding_dim": ParameterRule(int, 16, minimum=4, maximum=64),
            "hidden_dim": ParameterRule(int, 32, choices=(16, 32, 64)),
            "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
            "l2": ParameterRule(float, 1e-6, minimum=0.0, maximum=0.1),
            "patience": ParameterRule(int, 4, minimum=1, maximum=20),
            "batch_size": ParameterRule(int, 8192, choices=(2048, 4096, 8192, 16384)),
        },
        ensemble_members=3,
        objective="sequence_mlp",
    ),
    "pointwise_ensemble": ExperimentTemplate(
        name="pointwise_ensemble",
        description=(
            "Three consecutive-seed pointwise FMs averaged before validation "
            "scoring; intended to confirm replicated feature effects."
        ),
        parameters={
            "embedding_dim": ParameterRule(int, 16, minimum=1, maximum=128),
            "learning_rate": ParameterRule(float, 0.001, minimum=1e-6, maximum=0.1),
            "l2": ParameterRule(float, 1e-6, minimum=0.0, maximum=0.1),
            "patience": ParameterRule(int, 4, minimum=1, maximum=20),
            "batch_size": ParameterRule(int, 8192, choices=(2048, 4096, 8192, 16384)),
        },
        ensemble_members=3,
        objective="pointwise_bce",
    ),
    "bpr_hybrid": ExperimentTemplate(
        name="bpr_hybrid",
        description="One FM trained with within-user BPR plus auxiliary BCE.",
        parameters=_BPR_PARAMETERS,
    ),
    "bpr_ensemble": ExperimentTemplate(
        name="bpr_ensemble",
        description="Three consecutive-seed hybrid-BPR FMs averaged before scoring.",
        parameters={
            **_BPR_PARAMETERS,
            "popularity_weight": ParameterRule(
                float, 0.1, minimum=0.0, maximum=0.5
            ),
        },
        ensemble_members=3,
    ),
}


def get_template(name: str) -> ExperimentTemplate:
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        raise TemplateValidationError(
            f"unknown experiment template {name!r}; choose one of {sorted(TEMPLATES)}"
        ) from exc
