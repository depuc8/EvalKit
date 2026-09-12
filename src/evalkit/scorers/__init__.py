"""Built-in scorers and the registry used by the CLI."""

from __future__ import annotations

from evalkit.scorers.accuracy import AccuracyMode, AccuracyScorer
from evalkit.scorers.base import Score, ScoredRow, Scorer
from evalkit.scorers.groundedness import GroundednessScorer
from evalkit.scorers.performance import LatencyCostScorer, percentile
from evalkit.scorers.safety import DEFAULT_RULES, Finding, Rule, SafetyScorer
from evalkit.scorers.tools import ToolUseScorer, validate_args

BUILTIN_SCORERS: dict[str, type[Scorer]] = {
    "accuracy": AccuracyScorer,
    "groundedness": GroundednessScorer,
    "safety": SafetyScorer,
    "latency_cost": LatencyCostScorer,
    "tool_use": ToolUseScorer,
}

__all__ = [
    "BUILTIN_SCORERS",
    "DEFAULT_RULES",
    "AccuracyMode",
    "AccuracyScorer",
    "Finding",
    "GroundednessScorer",
    "LatencyCostScorer",
    "Rule",
    "SafetyScorer",
    "Score",
    "ScoredRow",
    "Scorer",
    "ToolUseScorer",
    "percentile",
    "validate_args",
]
