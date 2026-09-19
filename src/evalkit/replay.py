"""Replay a recorded run, optionally swapping the prompt or model at one LLM step, and diff.

How it works
------------
A replay re-runs the agent on the original input with a :class:`ReplayPlan` active. Every
LLM call made through a replay-aware model wrapper (see
:class:`evalkit.adapters.langgraph.ReplayableLLM`) asks the plan what to do:

* calls *before* the chosen step are answered from the recorded outputs (deterministic,
  free), so the run reaches the chosen step in the same state as before;
* the chosen step is executed live with the overridden prompt and/or model;
* later calls run live, so downstream effects of the change show up.

Tools always run live. Without an override the whole run is served from the recording.
"""

from __future__ import annotations

import difflib
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from evalkit.adapters.base import AgentAdapter
from evalkit.capture import StepListener, TraceExporter, TraceRecorder, use_recorder
from evalkit.trace import LLMCall, Step, Trace, TraceStore


class ReplayError(RuntimeError):
    """The replay request is invalid."""


@dataclass(frozen=True)
class StepOverride:
    """Changes applied to one LLM call.

    ``prompt`` replaces the whole prompt with a single user message; ``system`` replaces (or
    adds) the system message and keeps the rest; ``model`` runs the call on another model.
    """

    model: str | None = None
    prompt: str | None = None
    system: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.model is None and self.prompt is None and self.system is None

    def describe(self) -> str:
        parts = []
        if self.model:
            parts.append(f"model={self.model}")
        if self.system is not None:
            parts.append("system prompt replaced")
        if self.prompt is not None:
            parts.append("prompt replaced")
        return ", ".join(parts) or "no changes"


@dataclass(frozen=True)
class Decision:
    """What a replay-aware LLM wrapper should do for its next call."""

    index: int
    cached: LLMCall | None = None
    override: StepOverride | None = None


class ReplayPlan:
    """Decides, per LLM call (by order of occurrence), whether to reuse, override or run live."""

    def __init__(
        self,
        recorded: Sequence[LLMCall],
        *,
        override_index: int | None = None,
        override: StepOverride | None = None,
    ) -> None:
        self.recorded = list(recorded)
        self.override_index = override_index
        self.override = override
        self.override_applied = False
        self.calls_seen = 0
        self._lock = threading.Lock()

    def next_decision(self) -> Decision:
        with self._lock:
            index = self.calls_seen
            self.calls_seen += 1
            if self.override_index is not None:
                if index < self.override_index and index < len(self.recorded):
                    return Decision(index, cached=self.recorded[index])
                if index == self.override_index and self.override is not None:
                    self.override_applied = True
                    return Decision(index, override=self.override)
                return Decision(index)
            if index < len(self.recorded):
                return Decision(index, cached=self.recorded[index])
            return Decision(index)


_current_plan: ContextVar[ReplayPlan | None] = ContextVar("evalkit_replay_plan", default=None)


def current_plan() -> ReplayPlan | None:
    return _current_plan.get()


@contextmanager
def use_plan(plan: ReplayPlan) -> Iterator[ReplayPlan]:
    token = _current_plan.set(plan)
    try:
        yield plan
    finally:
        _current_plan.reset(token)


def llm_ordinal(trace: Trace, step_index: int) -> int:
    """Map a step's position in ``trace.steps`` to its order among LLM calls."""
    if not 0 <= step_index < len(trace.steps):
        raise ReplayError(f"step {step_index} out of range (trace has {len(trace.steps)} steps)")
    if trace.steps[step_index].llm is None:
        step = trace.steps[step_index]
        raise ReplayError(
            f"step {step_index} ({step.kind}:{step.name}) is not an LLM call; "
            f"LLM steps are {[i for i, s in enumerate(trace.steps) if s.llm is not None]}"
        )
    return sum(1 for s in trace.steps[:step_index] if s.llm is not None)


# -- diff -----------------------------------------------------------------------------
class StepDiff(BaseModel):
    status: Literal["same", "changed", "added", "removed"]
    label: str
    changes: list[str] = Field(default_factory=list)
    text_diff: str = ""


class TraceDiff(BaseModel):
    output_changed: bool
    original_output: str
    new_output: str
    output_diff: str
    steps: list[StepDiff]
    latency_delta_s: float
    tokens_delta: int

    @property
    def changed(self) -> bool:
        return self.output_changed or any(s.status != "same" for s in self.steps)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _udiff(a: str, b: str, fromfile: str, tofile: str) -> str:
    lines = difflib.unified_diff(
        a.splitlines(), b.splitlines(), fromfile=fromfile, tofile=tofile, lineterm="", n=1
    )
    return "\n".join(lines)


def _label(step: Step) -> str:
    return f"{step.kind}:{step.name}"


def _signature(step: Step) -> str:
    """Alignment key: LLM steps match each other even when the model name changed."""
    return "llm" if step.llm is not None else _label(step)


def _compare_steps(a: Step, b: Step) -> tuple[list[str], str]:
    changes: list[str] = []
    diffs: list[str] = []
    if a.llm is not None and b.llm is not None:
        if a.llm.model != b.llm.model:
            changes.append(f"model {a.llm.model} -> {b.llm.model}")
        if a.llm.prompt_text != b.llm.prompt_text:
            changes.append("prompt changed")
            diffs.append(
                _udiff(a.llm.prompt_text, b.llm.prompt_text, "prompt (original)", "prompt (new)")
            )
        if a.llm.output != b.llm.output:
            changes.append("output changed")
            diffs.append(_udiff(a.llm.output, b.llm.output, "output (original)", "output (new)"))
    elif a.tool is not None and b.tool is not None:
        if a.tool.args != b.tool.args:
            changes.append("arguments changed")
            diffs.append(
                _udiff(_text(a.tool.args), _text(b.tool.args), "args (original)", "args (new)")
            )
        if a.tool.result != b.tool.result or a.tool.error != b.tool.error:
            changes.append("result changed")
            diffs.append(
                _udiff(
                    _text(a.tool.result), _text(b.tool.result), "result (original)", "result (new)"
                )
            )
    elif a.outputs != b.outputs:
        changes.append("output changed")
        diffs.append(
            _udiff(_text(a.outputs), _text(b.outputs), "output (original)", "output (new)")
        )
    return changes, "\n".join(d for d in diffs if d)


def diff_traces(original: Trace, new: Trace) -> TraceDiff:
    """Align the two runs' steps by (kind, name) and report what changed."""
    sig_a, sig_b = [_signature(s) for s in original.steps], [_signature(s) for s in new.steps]
    steps: list[StepDiff] = []
    matcher = difflib.SequenceMatcher(None, sig_a, sig_b, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            for sa, sb in zip(original.steps[a0:a1], new.steps[b0:b1], strict=True):
                changes, text = _compare_steps(sa, sb)
                steps.append(
                    StepDiff(
                        status="changed" if changes else "same",
                        label=_label(sa),
                        changes=changes,
                        text_diff=text,
                    )
                )
            continue
        for sa in original.steps[a0:a1]:
            steps.append(StepDiff(status="removed", label=_label(sa)))
        for sb in new.steps[b0:b1]:
            steps.append(StepDiff(status="added", label=_label(sb)))
    out_a, out_b = original.answer, new.answer
    return TraceDiff(
        output_changed=out_a != out_b,
        original_output=out_a,
        new_output=out_b,
        output_diff=_udiff(out_a, out_b, "answer (original)", "answer (new)"),
        steps=steps,
        latency_delta_s=new.latency_s - original.latency_s,
        tokens_delta=new.usage.total_tokens - original.usage.total_tokens,
    )


def render_diff(diff: TraceDiff, *, color: bool = False) -> str:
    """Human-readable report of a :class:`TraceDiff`."""
    import click

    def paint(text: str, fg: str | None) -> str:
        return click.style(text, fg=fg) if color and fg else text

    marks = {
        "same": ("=", None),
        "changed": ("~", "yellow"),
        "added": ("+", "green"),
        "removed": ("-", "red"),
    }
    lines = ["Steps:"]
    for i, step in enumerate(diff.steps):
        mark, fg = marks[step.status]
        note = f"  ({'; '.join(step.changes)})" if step.changes else ""
        lines.append(paint(f"  {mark} {i:>2} {step.label}{note}", fg))
        for dl in step.text_diff.splitlines():
            dl_fg = (
                "green"
                if dl.startswith("+") and not dl.startswith("+++")
                else ("red" if dl.startswith("-") and not dl.startswith("---") else None)
            )
            lines.append("       " + paint(dl, dl_fg))
    lines.append("")
    if diff.output_changed:
        lines.append(paint("Final answer CHANGED", "yellow"))
        lines.extend(
            "  "
            + paint(
                dl,
                "green"
                if dl.startswith("+") and not dl.startswith("+++")
                else ("red" if dl.startswith("-") and not dl.startswith("---") else None),
            )
            for dl in diff.output_diff.splitlines()
        )
    else:
        lines.append("Final answer unchanged")
    lines.append(
        f"Latency delta: {diff.latency_delta_s:+.3f}s   Token delta: {diff.tokens_delta:+d}"
    )
    return "\n".join(lines)


def describe_steps(trace: Trace) -> str:
    """One line per step with its index, for choosing where to override."""
    rows = []
    for i, s in enumerate(trace.steps):
        extra = ""
        if s.llm is not None:
            extra = f"  model={s.llm.model} tokens={s.llm.usage.total_tokens}"
        elif s.tool is not None:
            extra = f"  args={s.tool.args}"
        rows.append(f"{i:>3}  {s.kind:<9} {s.name:<28} {s.latency_s * 1000:8.1f} ms{extra}")
    return "\n".join(rows)


# -- running a replay -------------------------------------------------------------------
class ReplayResult(BaseModel):
    original: Trace
    replayed: Trace
    diff: TraceDiff
    override_applied: bool


def replay_trace(
    trace: Trace,
    adapter: AgentAdapter,
    *,
    step: int | None = None,
    override: StepOverride | None = None,
    on_step: StepListener | None = None,
    store: TraceStore | None = None,
    exporters: Sequence[TraceExporter] = (),
) -> ReplayResult:
    """Re-run ``trace`` on ``adapter``; see the module docstring for semantics."""
    if (step is None) != (override is None or override.is_empty):
        raise ReplayError("pass both a step and a non-empty override, or neither")
    index = llm_ordinal(trace, step) if step is not None else None
    plan = ReplayPlan(trace.llm_calls, override_index=index, override=override)
    recorder = TraceRecorder(
        agent=trace.agent,
        input=trace.input,
        case_id=trace.case_id,
        context_tools=adapter.context_tools,
        metadata={
            "replay_of": trace.trace_id,
            "replay_step": step,
            "replay_override": override.describe() if override else "none (cached replay)",
        },
        on_step=on_step,
    )
    output: Any = None
    error: str | None = None
    try:
        with use_recorder(recorder), use_plan(plan):
            output = adapter.invoke(trace.input, recorder)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    new = recorder.finish(output=output, error=error)
    if override is not None and not plan.override_applied:
        raise ReplayError(
            "the override was never applied: the agent did not make an LLM call through a "
            "replay-aware wrapper (wrap your model with evalkit.adapters.langgraph.replayable)"
        )
    if store is not None:
        store.append(new)
    for exporter in exporters:
        exporter(new)
    return ReplayResult(
        original=trace,
        replayed=new,
        diff=diff_traces(trace, new),
        override_applied=plan.override_applied,
    )
