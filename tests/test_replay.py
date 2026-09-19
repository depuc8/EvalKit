from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalkit.adapters import AgentAdapter
from evalkit.capture import TraceRecorder, current_recorder
from evalkit.replay import (
    ReplayError,
    ReplayPlan,
    StepOverride,
    current_plan,
    describe_steps,
    diff_traces,
    llm_ordinal,
    render_diff,
    replay_trace,
    use_plan,
)
from evalkit.trace import ToolCall, TraceStore
from tests.helpers import make_llm, make_trace


class ScriptedAdapter(AgentAdapter):
    """Two LLM calls with a tool in between; replay-aware like ReplayableLLM."""

    name = "scripted"

    def __init__(self, answers: tuple[str, str] = ("draft", "final")) -> None:
        self.answers = answers
        self.live_calls: list[tuple[int, str | None]] = []

    def _llm(self, rec: TraceRecorder, prompt: str, live: str) -> str:
        plan = current_plan()
        decision = plan.next_decision() if plan else None
        if decision is not None and decision.cached is not None:
            rec.record_llm(model="m", prompt=prompt, output=decision.cached.output, cached=True)
            return decision.cached.output
        model = decision.override.model if decision and decision.override else "m"
        text = (
            decision.override.prompt
            if decision and decision.override and decision.override.prompt
            else live
        )
        self.live_calls.append((decision.index if decision else -1, model))
        rec.record_llm(
            model=model or "m", prompt=prompt, output=text, input_tokens=2, output_tokens=1
        )
        return text

    def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
        rec = current_recorder() or recorder
        draft = self._llm(rec, "p1", self.answers[0])
        rec.record_tool("fetch", {"q": draft}, result="data")
        return self._llm(rec, "p2", self.answers[1])


def original_trace() -> Any:
    adapter = ScriptedAdapter()
    rec = TraceRecorder(agent="scripted", input="question", case_id="c9")
    from evalkit.capture import use_recorder

    with use_recorder(rec):
        out = adapter.invoke("question", rec)
    return rec.finish(output=out)


def test_plan_decisions() -> None:
    recorded = [make_llm(), make_llm()]
    plan = ReplayPlan(recorded, override_index=1, override=StepOverride(model="x"))
    first, second, third = plan.next_decision(), plan.next_decision(), plan.next_decision()
    assert first.cached is recorded[0]
    assert second.override == StepOverride(model="x") and second.cached is None
    assert third.cached is None and third.override is None
    assert plan.override_applied and plan.calls_seen == 3
    full = ReplayPlan(recorded)
    assert [full.next_decision().cached is not None for _ in range(3)] == [True, True, False]
    beyond = ReplayPlan([], override_index=0, override=StepOverride(prompt="p"))
    assert beyond.next_decision().override is not None


def test_use_plan_scopes_context() -> None:
    assert current_plan() is None
    plan = ReplayPlan([])
    with use_plan(plan):
        assert current_plan() is plan
    assert current_plan() is None


def test_override_describe() -> None:
    assert StepOverride().is_empty and StepOverride().describe() == "no changes"
    text = StepOverride(model="m", system="s", prompt="p").describe()
    assert "model=m" in text and "system prompt replaced" in text and "prompt replaced" in text


def test_llm_ordinal() -> None:
    trace = original_trace()
    assert [s.kind for s in trace.steps] == ["llm", "tool", "llm"]
    assert llm_ordinal(trace, 0) == 0 and llm_ordinal(trace, 2) == 1
    with pytest.raises(ReplayError, match="not an LLM call"):
        llm_ordinal(trace, 1)
    with pytest.raises(ReplayError, match="out of range"):
        llm_ordinal(trace, 9)


def test_replay_without_override_serves_everything_from_cache() -> None:
    trace = original_trace()
    adapter = ScriptedAdapter(answers=("changed", "changed"))
    result = replay_trace(trace, adapter)
    assert adapter.live_calls == [] and result.replayed.output == trace.output
    assert not result.diff.changed and result.override_applied is False
    assert result.replayed.metadata["replay_of"] == trace.trace_id
    assert all(s.llm.cached for s in result.replayed.llm_steps if s.llm)


def test_replay_with_override_runs_live_from_the_chosen_step() -> None:
    trace = original_trace()
    adapter = ScriptedAdapter(answers=("draft", "second try"))
    exported, steps = [], []
    result = replay_trace(
        trace,
        adapter,
        step=2,
        override=StepOverride(model="better"),
        on_step=steps.append,
        exporters=[exported.append],
    )
    assert adapter.live_calls == [(1, "better")]
    assert result.override_applied and result.diff.output_changed
    assert result.replayed.output == "second try" and exported and len(steps) == 3
    kinds = {s.status for s in result.diff.steps}
    assert "changed" in kinds
    assert any("model m -> better" in c for s in result.diff.steps for c in s.changes)


def test_replay_saves_trace(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "r.jsonl")
    replay_trace(original_trace(), ScriptedAdapter(), store=store)
    assert len(store.read_all()) == 1


def test_replay_validation_errors() -> None:
    trace = original_trace()
    with pytest.raises(ReplayError, match="both a step and a non-empty override"):
        replay_trace(trace, ScriptedAdapter(), step=0)
    with pytest.raises(ReplayError, match="both a step and a non-empty override"):
        replay_trace(trace, ScriptedAdapter(), override=StepOverride(model="x"))
    with pytest.raises(ReplayError, match="both a step and a non-empty override"):
        replay_trace(trace, ScriptedAdapter(), step=0, override=StepOverride())


def test_replay_detects_override_that_never_applied() -> None:
    class Oblivious(AgentAdapter):
        def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
            recorder.record_llm(model="m", prompt="p", output="o")
            return "o"

    trace = make_trace("o", llm=[make_llm()])
    with pytest.raises(ReplayError, match="never applied"):
        replay_trace(trace, Oblivious(), step=0, override=StepOverride(model="x"))


def test_replay_records_agent_failure() -> None:
    class Broken(AgentAdapter):
        def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
            raise RuntimeError("replay boom")

    result = replay_trace(make_trace("o", llm=[make_llm()]), Broken())
    assert result.replayed.error == "RuntimeError: replay boom"


# ---- diff ------------------------------------------------------------------------------


def test_diff_identical_and_changed() -> None:
    a = make_trace(
        "same", llm=[make_llm(out=1)], tools=[ToolCall(name="t", args={"x": 1}, result=1)]
    )
    assert not diff_traces(a, a).changed
    b = make_trace(
        "different", llm=[make_llm(out=5)], tools=[ToolCall(name="t", args={"x": 2}, result=2)]
    )
    b.steps[0].llm.output = "new text"  # type: ignore[union-attr]
    b.steps[0].llm.prompt = [{"role": "user", "content": "other"}]  # type: ignore[union-attr]
    diff = diff_traces(a, b)
    assert diff.output_changed and diff.changed
    changes = [c for s in diff.steps for c in s.changes]
    assert {"prompt changed", "output changed", "arguments changed", "result changed"} <= set(
        changes
    )
    assert diff.tokens_delta == 4


def test_diff_added_and_removed_steps() -> None:
    a = make_trace("x", tools=[ToolCall(name="old")])
    b = make_trace("x", tools=[ToolCall(name="new")])
    statuses = sorted(s.status for s in diff_traces(a, b).steps)
    assert statuses == ["added", "removed"]


def test_diff_compares_plain_step_outputs() -> None:
    a, b = make_trace("x", docs=["d"]), make_trace("x", docs=["d"])
    a.steps[0].outputs, b.steps[0].outputs = "one", "two"
    assert diff_traces(a, b).steps[0].changes == ["output changed"]


def test_render_diff_plain_and_colored() -> None:
    a = make_trace("old answer", llm=[make_llm()], tools=[ToolCall(name="gone")])
    b = make_trace("new answer", llm=[make_llm("other")])
    b.steps[0].llm.output = "different"  # type: ignore[union-attr]
    diff = diff_traces(a, b)
    plain = render_diff(diff)
    assert "Final answer CHANGED" in plain and "-old answer" in plain and "+new answer" in plain
    assert "Latency delta" in plain and "\x1b[" not in plain
    assert "\x1b[" in render_diff(diff, color=True)
    assert "Final answer unchanged" in render_diff(diff_traces(a, a))


def test_describe_steps() -> None:
    text = describe_steps(original_trace())
    assert "llm" in text and "fetch" in text and "args=" in text
    assert describe_steps(make_trace("x", docs=["d"])).strip().startswith("0")
