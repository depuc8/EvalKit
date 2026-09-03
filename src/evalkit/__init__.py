"""EvalKit: evaluate and debug LLM agents.

Public names are imported lazily (PEP 562) so that ``import evalkit`` - which pytest does at
startup through the plugin entry point - stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from evalkit._version import __version__

if TYPE_CHECKING:
    from evalkit.adapters.base import AgentAdapter, FunctionAdapter, adapt, register_adapter
    from evalkit.assertions import EvalAssertionError, assert_eval
    from evalkit.baseline import compare, load_baseline, save_baseline
    from evalkit.capture import TraceRecorder, current_recorder, trace_run, traced
    from evalkit.dataset import Case, ExpectedTool, load_dataset
    from evalkit.pricing import ModelPrice, PricingConfig
    from evalkit.registry import build_scorers
    from evalkit.replay import StepOverride, diff_traces, replay_trace
    from evalkit.report import build_report, render_html, render_terminal
    from evalkit.runner import CaseResult, EvalResults, run_eval, run_eval_async
    from evalkit.scorers import (
        AccuracyScorer,
        GroundednessScorer,
        LatencyCostScorer,
        SafetyScorer,
        Score,
        Scorer,
        ToolUseScorer,
    )
    from evalkit.trace import LLMCall, Step, TokenUsage, ToolCall, Trace, TraceStore

_EXPORTS: dict[str, str] = {
    "AgentAdapter": "evalkit.adapters.base",
    "FunctionAdapter": "evalkit.adapters.base",
    "adapt": "evalkit.adapters.base",
    "register_adapter": "evalkit.adapters.base",
    "EvalAssertionError": "evalkit.assertions",
    "assert_eval": "evalkit.assertions",
    "compare": "evalkit.baseline",
    "load_baseline": "evalkit.baseline",
    "save_baseline": "evalkit.baseline",
    "TraceRecorder": "evalkit.capture",
    "current_recorder": "evalkit.capture",
    "trace_run": "evalkit.capture",
    "traced": "evalkit.capture",
    "Case": "evalkit.dataset",
    "ExpectedTool": "evalkit.dataset",
    "load_dataset": "evalkit.dataset",
    "ModelPrice": "evalkit.pricing",
    "PricingConfig": "evalkit.pricing",
    "build_scorers": "evalkit.registry",
    "StepOverride": "evalkit.replay",
    "diff_traces": "evalkit.replay",
    "replay_trace": "evalkit.replay",
    "build_report": "evalkit.report",
    "render_html": "evalkit.report",
    "render_terminal": "evalkit.report",
    "CaseResult": "evalkit.runner",
    "EvalResults": "evalkit.runner",
    "run_eval": "evalkit.runner",
    "run_eval_async": "evalkit.runner",
    "AccuracyScorer": "evalkit.scorers",
    "GroundednessScorer": "evalkit.scorers",
    "LatencyCostScorer": "evalkit.scorers",
    "SafetyScorer": "evalkit.scorers",
    "Score": "evalkit.scorers",
    "Scorer": "evalkit.scorers",
    "ToolUseScorer": "evalkit.scorers",
    "LLMCall": "evalkit.trace",
    "Step": "evalkit.trace",
    "TokenUsage": "evalkit.trace",
    "ToolCall": "evalkit.trace",
    "Trace": "evalkit.trace",
    "TraceStore": "evalkit.trace",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'evalkit' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
