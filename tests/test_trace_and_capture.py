from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pytest
from pydantic import BaseModel

from evalkit.capture import (
    TraceRecorder,
    current_recorder,
    normalize_documents,
    to_jsonable,
    trace_run,
    traced,
)
from evalkit.trace import LLMCall, TokenUsage, ToolCall, TraceStore
from tests.helpers import make_llm, make_trace


def test_token_usage_adds_and_totals() -> None:
    total = TokenUsage(input_tokens=1, output_tokens=2) + TokenUsage(
        input_tokens=3, output_tokens=4
    )
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (4, 6, 10)


def test_llm_call_prompt_text() -> None:
    call = LLMCall(prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert call.prompt_text == "[system] s\n[user] u"


def test_trace_properties() -> None:
    trace = make_trace(
        "answer",
        llm=[make_llm(inp=5, out=1), make_llm(inp=7, out=2)],
        tools=[ToolCall(name="t", args={"a": 1})],
        docs=["doc one"],
    )
    assert trace.answer == "answer"
    assert trace.usage.total_tokens == 15
    assert len(trace.llm_calls) == 2 and len(trace.llm_steps) == 2
    assert [t.name for t in trace.tool_calls] == ["t"]
    assert trace.retrieved_context() == ["doc one"]


def test_trace_answer_handles_none_and_non_strings() -> None:
    assert make_trace(None).answer == ""
    assert make_trace({"a": 1}).answer == "{'a': 1}"


def test_store_roundtrip_and_prefix_lookup(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "sub" / "t.jsonl")
    assert store.read_all() == []
    a, b = make_trace("a"), make_trace("b")
    store.append(a)
    store.append(b)
    assert [t.answer for t in store.read_all()] == ["a", "b"]
    assert store.get(a.trace_id).answer == "a"
    assert store.get(a.trace_id[:10]).trace_id == a.trace_id
    with pytest.raises(KeyError, match="no trace"):
        store.get("zzzz")
    with pytest.raises(KeyError, match="ambiguous"):
        store.get("")


def test_to_jsonable_handles_many_types() -> None:
    class Model(BaseModel):
        x: int

    @dataclasses.dataclass
    class DC:
        y: int

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    value = {
        "m": Model(x=1),
        "d": DC(y=2),
        "t": (1, 2),
        "s": {3},
        "dt": dt.date(2024, 1, 2),
        "p": Path("/x"),
        "b": b"hi",
        "o": Odd(),
        1: None,
    }
    out = to_jsonable(value)
    assert out["m"] == {"x": 1} and out["d"] == {"y": 2}
    assert out["t"] == [1, 2] and out["s"] == [3]
    assert out["dt"] == "2024-01-02" and out["p"] == "/x" and out["b"] == "hi"
    assert out["o"] == "<odd>" and out["1"] is None


def test_to_jsonable_depth_guard() -> None:
    nested: list[object] = []
    cursor = nested
    for _ in range(30):
        inner: list[object] = []
        cursor.append(inner)
        cursor = inner
    assert "[" in str(to_jsonable(nested))


def test_normalize_documents() -> None:
    class Doc:
        page_content = "from attr"

    assert normalize_documents(None) == []
    assert normalize_documents("  ") == []
    assert normalize_documents("text") == ["text"]
    assert normalize_documents({"page_content": "pc"}) == ["pc"]
    assert normalize_documents({"other": 1}) == ["{'other': 1}"]
    assert normalize_documents(["a", {"text": "b"}]) == ["a", "b"]
    assert normalize_documents(Doc()) == ["from attr"]
    assert normalize_documents(42) == ["42"]


def test_recorder_records_steps_and_finishes() -> None:
    rec = TraceRecorder(agent="x", input={"q": 1}, case_id="c", trace_id="fixed")
    sid = rec.begin_step("node", "node", inputs={"a": 1})
    rec.end_step(sid, outputs="done")
    rec.record_llm(
        model="m",
        prompt="hi",
        output="yo",
        input_tokens=3,
        output_tokens=2,
        latency_s=0.5,
        time_to_first_token_s=0.1,
        tool_calls=[{"name": "t"}],
    )
    rec.record_tool("search", {"q": "x"}, result="r", latency_s=0.2)
    rec.record_tool("bad", {}, error="boom")
    trace = rec.finish(output="final")
    assert trace.trace_id == "fixed" and trace.output == "final" and trace.case_id == "c"
    kinds = {s.kind for s in trace.steps}
    assert kinds == {"node", "llm", "tool"}
    llm = trace.llm_steps[0]
    assert llm.latency_s >= 0.5 and llm.llm is not None and llm.llm.usage.total_tokens == 5
    assert rec.finish(output="again") is trace  # idempotent


def test_recorder_context_tools_capture_documents() -> None:
    rec = TraceRecorder(context_tools=["search"])
    rec.record_tool("search", {}, result=["doc a", "doc b"])
    rec.record_tool("other", {}, result="not context")
    rec.record_tool("search", {}, error="failed")
    assert rec.finish().retrieved_context() == ["doc a", "doc b"]


def test_recorder_first_token_and_unfinished_steps() -> None:
    rec = TraceRecorder()
    sid = rec.begin_step("llm", "llm")
    rec.mark_first_token(sid)
    rec.mark_first_token(sid)  # second call ignored
    rec.mark_first_token("unknown")
    rec.end_step(sid, llm=LLMCall(model="m"))
    rec.begin_step("dangling", "custom")
    trace = rec.finish()
    assert trace.llm_calls[0].time_to_first_token_s is not None
    dangling = next(s for s in trace.steps if s.name == "dangling")
    assert dangling.error == "step never completed"


def test_recorder_ignores_unknown_and_late_events() -> None:
    rec = TraceRecorder()
    assert rec.end_step("nope") is None
    sid = rec.begin_step("late")
    rec.finish()
    assert rec.end_step(sid) is None


def test_recorder_step_context_manager() -> None:
    seen = []
    rec = TraceRecorder(on_step=seen.append)
    with rec.step("ok", "custom", inputs=1) as handle:
        handle.outputs = "fine"
    with pytest.raises(ValueError, match="bad"), rec.step("boom"):
        raise ValueError("bad")
    trace = rec.finish()
    assert [s.name for s in seen] == ["ok", "boom"]
    boom = next(s for s in trace.steps if s.name == "boom")
    assert boom.error == "ValueError: bad"


def test_trace_run_stores_and_exports(tmp_path: Path) -> None:
    exported = []
    path = tmp_path / "t.jsonl"
    with trace_run(agent="a", input="q", store=path, exporters=[exported.append]) as rec:
        assert current_recorder() is rec
        rec.record_llm(model="m", prompt="p", output="o")
        rec.result = "answer"
    assert current_recorder() is None
    stored = TraceStore(path).read_all()
    assert stored[0].output == "answer" and exported[0].trace_id == stored[0].trace_id


def test_trace_run_records_exception() -> None:
    exported = []
    with pytest.raises(RuntimeError), trace_run(exporters=[exported.append]):
        raise RuntimeError("kaput")
    assert exported[0].error == "RuntimeError: kaput"


def test_traced_decorator_sync(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.jsonl")

    @traced(store=store)
    def agent(question: str, *, loud: bool = False) -> str:
        rec = current_recorder()
        assert rec is not None
        rec.record_tool("lookup", {"q": question}, result="data")
        return question.upper() if loud else question

    assert agent("hi", loud=True) == "HI"
    (trace,) = store.read_all()
    assert trace.agent == "agent" and trace.output == "HI"
    assert trace.input == {"question": "hi", "loud": True}
    assert trace.tool_calls[0].name == "lookup"


def test_traced_single_argument_input_is_unwrapped() -> None:
    exported = []

    @traced(agent="named", exporters=[exported.append])
    def agent(question: str) -> str:
        return "a"

    agent("just this")
    assert exported[0].input == "just this" and exported[0].agent == "named"


async def test_traced_decorator_async(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.jsonl")

    @traced(store=store)
    async def agent(question: str) -> str:
        return question[::-1]

    assert await agent("abc") == "cba"
    assert store.read_all()[0].output == "cba"


def test_traced_records_errors_and_rebinds_failure() -> None:
    exported = []

    @traced(exporters=[exported.append])
    def agent(a: int) -> int:
        raise ZeroDivisionError("nope")

    with pytest.raises(ZeroDivisionError):
        agent(1)
    assert exported[0].error.startswith("ZeroDivisionError")

    @traced(exporters=[exported.append])
    def takes_none() -> str:
        return "x"

    with pytest.raises(TypeError):
        takes_none(1)  # type: ignore[call-arg]  # bad call: input falls back to raw args
    assert exported[-1].input == {"args": [1], "kwargs": {}}
