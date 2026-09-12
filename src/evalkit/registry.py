"""Build scorer lists from names/specs (used by the CLI; handy in scripts too)."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from evalkit.judge import Judge
from evalkit.pricing import PricingConfig
from evalkit.scorers import (
    AccuracyMode,
    AccuracyScorer,
    GroundednessScorer,
    LatencyCostScorer,
    SafetyScorer,
    Scorer,
    ToolUseScorer,
)

DEFAULT_SCORERS = ("accuracy", "groundedness", "safety", "latency_cost", "tool_use")


class ScorerSpecError(ValueError):
    """An unknown scorer name or unloadable ``module:Class`` spec."""


def build_scorers(
    specs: Sequence[str] = DEFAULT_SCORERS,
    *,
    judge: Judge | None = None,
    accuracy_mode: AccuracyMode = "fuzzy",
    pricing: PricingConfig | None = None,
    max_latency_s: float | None = None,
    max_cost_usd: float | None = None,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> list[Scorer]:
    """Instantiate scorers by built-in name or ``module:Class`` (zero-argument constructor)."""
    scorers: list[Scorer] = []
    for spec in specs:
        if spec == "accuracy":
            scorers.append(AccuracyScorer(accuracy_mode, judge=judge))
        elif spec == "groundedness":
            scorers.append(GroundednessScorer(judge=judge))
        elif spec == "safety":
            scorers.append(SafetyScorer(judge=judge))
        elif spec == "latency_cost":
            scorers.append(
                LatencyCostScorer(
                    pricing=pricing, max_latency_s=max_latency_s, max_cost_usd=max_cost_usd
                )
            )
        elif spec == "tool_use":
            scorers.append(ToolUseScorer(schemas=tool_schemas))
        elif ":" in spec:
            scorers.append(_load_custom(spec))
        else:
            raise ScorerSpecError(
                f"unknown scorer {spec!r}; built-ins are {', '.join(DEFAULT_SCORERS)} "
                "or use 'module:Class' for a custom scorer"
            )
    return scorers


def _load_custom(spec: str) -> Scorer:
    module_name, _, attr = spec.partition(":")
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ScorerSpecError(f"cannot load scorer {spec!r}: {exc}") from exc
    try:
        instance = obj() if callable(obj) else obj
    except TypeError as exc:
        raise ScorerSpecError(
            f"cannot instantiate scorer {spec!r} without arguments: {exc}"
        ) from exc
    if not isinstance(instance, Scorer):
        raise ScorerSpecError(f"{spec!r} is not a Scorer subclass or instance")
    return instance
