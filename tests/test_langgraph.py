from __future__ import annotations

import asyncio
from typing import Annotated, Any, TypedDict

import pytest
from examples.rag_agent.fake_model import ExtractiveChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from evalkit.adapters import adapt
from evalkit.adapters.langgraph import (
    EvalKitCallback,
    LangGraphAdapter,
    ReplayableLLM,
    _apply_override,
    _text,
    replayable,
)
from evalkit.capture import TraceRecorder
from evalkit.dataset import Case
from evalkit.replay import ReplayError, ReplayPlan, StepOverride, use_plan
from evalkit.runner import run_eval
from evalkit.trace import LLMCall, TokenUsage


class State(TypedDict, total=False):
    question: str
    docs: list[str]
    answer: str


class Docs(BaseRetriever):
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return [Document(page_content=f"doc about {query}")]


@tool
def shout(text: str) -> str:
    """Upper-case text."""
    return text.upper()


@tool
def explode(text: str) -> str:
    """Always fails."""
    raise RuntimeError("tool failure")


def build(llm: Any = None, *, fail: bool = False) -> Any:
    model = llm or ExtractiveChatModel()

    def retrieve(state: State) -> State:
        docs = Docs().invoke(state["question"])
        shout.invoke({"text": state["question"]})
        if fail:
            explode.invoke({"text": "x"})
        return {"docs": [d.page_content for d in docs]}

    def generate(state: State) -> State:
        prompt = "Context:\n[1] " + " ".join(state["docs"]) + f"\n\nQuestion: {state['question']}"
        reply = model.invoke([SystemMessage("be brief"), HumanMessage(prompt)])
        return {"answer": str(reply.content)}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(name="demo")


def run(graph: Any, question: str = "What is the cloud plan?", **kw: Any) -> Any:
    adapter = LangGraphAdapter(graph, **kw)
    rec = TraceRecorder(agent=adapter.name)
    output = adapter.invoke(question, rec)
    return rec.finish(output=output)


def test_callback_captures_nodes_llm_tool_retriever() -> None:
    trace = run(build())
    assert trace.agent == "demo"
    by_kind: dict[str, list[str]] = {}
    for s in trace.steps:
        by_kind.setdefault(s.kind, []).append(s.name)
    assert by_kind["node"] == ["retrieve", "generate"]
    assert by_kind["tool"] == ["shout"] and by_kind["retriever"] == ["Docs"]
    llm = trace.llm_steps[0].llm
    assert llm is not None and llm.model == "acme-extractive-v1"
    assert llm.prompt[0] == {"role": "system", "content": "be brief"}
    assert llm.usage.total_tokens > 0 and llm.output
    assert trace.retrieved_context() == ["doc about What is the cloud plan?"]
    assert trace.tool_calls[0].args == {"text": "What is the cloud plan?"}
    assert trace.tool_calls[0].result == "WHAT IS THE CLOUD PLAN?"
    generate = next(s for s in trace.steps if s.name == "generate")
    llm_step = trace.llm_steps[0]
    assert llm_step.parent_id == generate.id  # nesting is preserved
    assert isinstance(trace.output, str) and "doc about" in trace.output


def test_streaming_mode_reports_time_to_first_token() -> None:
    assert run(build()).llm_calls[0].time_to_first_token_s is not None
    assert run(build(), measure_ttft=False).llm_calls[0].time_to_first_token_s is None


def test_tool_error_is_recorded() -> None:
    adapter = LangGraphAdapter(build(fail=True), measure_ttft=False)
    rec = TraceRecorder()
    with pytest.raises(RuntimeError, match="tool failure"):
        adapter.invoke("q", rec)
    trace = rec.finish(error="x")
    bad = next(t for t in trace.tool_calls if t.name == "explode")
    assert bad.error == "RuntimeError: tool failure"
    assert any(s.kind == "node" and s.error for s in trace.steps)


async def test_async_invoke_paths() -> None:
    for ttft in (True, False):
        adapter = LangGraphAdapter(build(), measure_ttft=ttft)
        rec = TraceRecorder()
        output = await adapter.ainvoke("async question", rec)
        assert "async question" in output and rec.finish().llm_steps


def test_input_and_output_mapping() -> None:
    class M(TypedDict, total=False):
        messages: Annotated[list[Any], add_messages]
        final: str

    def node(state: M) -> M:
        text = state["messages"][-1].content
        return {"messages": [AIMessage(f"heard {text}")], "final": "F"}

    g = StateGraph(M)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile()
    assert run(graph, "hello", input_key="messages", measure_ttft=False).output == "heard hello"
    assert run(graph, "hello", input_key="messages", output_key="final").output == "F"
    adapter = LangGraphAdapter(graph, input_key="messages")
    assert adapter.extract_answer({"other": 1}) == {"other": 1}
    assert adapter.extract_answer("raw") == "raw"
    assert adapter.extract_answer(AIMessage("m")) == "m"
    assert adapter.extract_answer({"messages": ["plain"]}) == "plain"
    dict_in = LangGraphAdapter(graph)._payload({"messages": [("user", "x")]})
    assert dict_in == {"messages": [("user", "x")]}


def test_adapt_detects_langgraph_and_context_tools() -> None:
    adapter = adapt(build())
    assert isinstance(adapter, LangGraphAdapter) and adapter.name == "demo"
    assert LangGraphAdapter(build(), context_tools=["shout"]).context_tools == {"shout"}
    cfg = LangGraphAdapter(build(), config={"callbacks": [], "tags": ["t"]})
    assert cfg._config(TraceRecorder())["tags"] == ["t"]


def test_context_tools_feed_documents() -> None:
    adapter = LangGraphAdapter(build(), context_tools=["shout"], measure_ttft=False)
    rec = TraceRecorder(context_tools=adapter.context_tools)
    adapter.invoke("hi", rec)
    assert "HI" in rec.finish().retrieved_context()


def test_runs_through_the_evaluation_runner() -> None:
    cases = [Case(id="a", input="alpha"), Case(id="b", input="beta")]
    results = run_eval(cases, LangGraphAdapter(build()), [], concurrency=2)
    assert all(c.trace.error is None and c.trace.steps for c in results.cases)


def test_callback_handles_plain_llm_and_errors() -> None:
    from uuid import uuid4

    rec = TraceRecorder()
    cb = EvalKitCallback(rec)
    run_id = uuid4()
    cb.on_llm_start({"name": "plain"}, ["a prompt"], run_id=run_id)
    cb.on_llm_new_token("t", run_id=run_id)

    class Gen:
        text = "completion"
        message = None

    class Resp:
        generations = [[Gen()]]
        llm_output = {"token_usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    cb.on_llm_end(Resp(), run_id=run_id)
    bad = uuid4()
    cb.on_llm_start(None, ["p"], run_id=bad, metadata={"ls_model_name": "named"})
    cb.on_llm_error(ValueError("nope"), run_id=bad)
    chain = uuid4()
    cb.on_chain_start({"name": "n"}, {}, run_id=chain, metadata={"langgraph_node": "n"}, name="n")
    cb.on_chain_error(KeyError("k"), run_id=chain)
    retr = uuid4()
    cb.on_retriever_start({"name": "R"}, "q", run_id=retr)
    cb.on_retriever_error(OSError("net"), run_id=retr)
    tool_run = uuid4()
    cb.on_tool_start({"name": "t"}, "raw input", run_id=tool_run)
    cb.on_tool_end(AIMessage("tool message"), run_id=tool_run)
    # events for runs that were never recorded are ignored
    stray = uuid4()
    for fn in (cb.on_chain_end, cb.on_llm_end, cb.on_tool_end, cb.on_retriever_end):
        fn(None, run_id=stray)  # type: ignore[operator]
    for fn in (cb.on_chain_error, cb.on_llm_error, cb.on_tool_error, cb.on_retriever_error):
        fn(ValueError(), run_id=stray)  # type: ignore[operator]
    cb.on_llm_new_token("t", run_id=stray)
    trace = rec.finish()
    first = trace.llm_steps[0].llm
    assert first is not None and first.usage == TokenUsage(input_tokens=4, output_tokens=2)
    assert first.output == "completion" and first.prompt[0]["content"] == "a prompt"
    errored = [s for s in trace.steps if s.error]
    assert {s.kind for s in errored} == {"llm", "node", "retriever"}
    assert trace.tool_calls[0].args == {"input": "raw input"}
    assert trace.tool_calls[0].result == "tool message"


def test_text_helper() -> None:
    assert _text("s") == "s"
    assert _text([{"text": "a"}, "b", {"x": 1}]) == "ab"
    assert _text(5) == "5"


# ---- replayable LLM ----------------------------------------------------------------------


def recorded(output: str, model: str = "acme-extractive-v1") -> LLMCall:
    return LLMCall(
        model=model,
        prompt=[{"role": "user", "content": "p"}],
        output=output,
        usage=TokenUsage(input_tokens=3, output_tokens=1),
        tool_calls=[{"name": "t", "args": {"a": 1}, "id": "1"}],
    )


def test_replayable_is_transparent_without_a_plan() -> None:
    llm = replayable(ExtractiveChatModel())
    out = llm.invoke(
        [SystemMessage("s"), HumanMessage("Context:\n[1] Cats purr.\n\nQuestion: cats?")]
    )
    assert "purr" in str(out.content)
    assert isinstance(llm, ReplayableLLM)


def test_replayable_serves_cache_then_applies_override() -> None:
    llm = replayable(ExtractiveChatModel(), factory=lambda n: ExtractiveChatModel(model_name=n))
    plan = ReplayPlan(
        [recorded("cached answer")],
        override_index=1,
        override=StepOverride(model="acme-extractive-v2", system="untrusted"),
    )
    prompt = "Context:\n[1] Dogs bark loudly. Dogs also dig.\n\nQuestion: do dogs bark?"
    rec = TraceRecorder()
    from evalkit.capture import use_recorder

    with use_recorder(rec), use_plan(plan):
        first = llm.invoke([HumanMessage("ignored")])
        second = llm.invoke([HumanMessage(prompt)])
        third = llm.invoke([HumanMessage(prompt)])
    assert first.content == "cached answer" and first.tool_calls[0]["name"] == "t"
    assert first.usage_metadata["total_tokens"] == 4
    assert "bark" in str(second.content) and plan.override_applied
    assert "bark" in str(third.content)
    cached_steps = [s for s in rec.finish().llm_steps if s.llm and s.llm.cached]
    assert len(cached_steps) == 1


def test_replayable_model_swap_requires_factory() -> None:
    llm = replayable(ExtractiveChatModel())
    plan = ReplayPlan([], override_index=0, override=StepOverride(model="other"))
    with use_plan(plan), pytest.raises(ReplayError, match="factory"):
        llm.invoke([HumanMessage("q")])


def test_replayable_bind_tools_survives_model_swap() -> None:
    class Bindable(ExtractiveChatModel):
        bound: list[Any] = []

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            clone = self.model_copy(update={"bound": list(tools)})
            return clone

    llm = replayable(Bindable(), factory=lambda n: Bindable(model_name=n)).bind_tools(["tool-a"])
    assert isinstance(llm, ReplayableLLM) and llm._model.bound == ["tool-a"]  # type: ignore[attr-defined]
    swapped = llm._swapped("v2")
    assert swapped.bound == ["tool-a"] and swapped.model_name == "v2"


def test_replayable_without_recorder_still_returns_cached() -> None:
    llm = replayable(ExtractiveChatModel())
    with use_plan(ReplayPlan([recorded("x")])):
        assert llm.invoke("anything").content == "x"


async def test_replayable_ainvoke() -> None:
    llm = replayable(ExtractiveChatModel())
    with use_plan(ReplayPlan([recorded("from cache")])):
        out = await llm.ainvoke("q")
    assert out.content == "from cache"
    assert asyncio.get_running_loop() is not None


def test_apply_override_variants() -> None:
    msgs = [SystemMessage("old"), HumanMessage("hi")]
    assert _apply_override(msgs, StepOverride(prompt="new"))[0].content == "new"
    replaced = _apply_override(msgs, StepOverride(system="fresh"))
    assert [m.content for m in replaced] == ["fresh", "hi"]
    assert _apply_override(msgs, StepOverride(model="m")) is msgs
