"""Builders shared by the test modules."""

from __future__ import annotations

from typing import Any

from evalkit.dataset import Case
from evalkit.trace import LLMCall, Step, TokenUsage, ToolCall, Trace


def make_trace(
    answer: Any = "ok",
    *,
    latency: float = 0.1,
    llm: list[LLMCall] | None = None,
    tools: list[ToolCall] | None = None,
    docs: list[str] | None = None,
    error: str | None = None,
    case_id: str | None = "c1",
    input: Any = "question",
) -> Trace:
    steps: list[Step] = []
    for i, call in enumerate(llm or []):
        steps.append(
            Step(
                name=call.model or "llm", kind="llm", llm=call, latency_s=0.05, start_time=100.0 + i
            )
        )
    for i, tool in enumerate(tools or []):
        steps.append(Step(name=tool.name, kind="tool", tool=tool, start_time=200.0 + i))
    if docs:
        steps.append(Step(name="retriever", kind="retriever", documents=docs, start_time=50.0))
    return Trace(
        case_id=case_id, input=input, output=answer, latency_s=latency, steps=steps, error=error
    )


def make_llm(
    model: str = "m", out: int = 10, inp: int = 20, ttft: float | None = None, cached: bool = False
) -> LLMCall:
    return LLMCall(
        model=model,
        prompt=[{"role": "user", "content": "hi"}],
        output="hello",
        usage=TokenUsage(input_tokens=inp, output_tokens=out),
        time_to_first_token_s=ttft,
        cached=cached,
    )


def make_case(**kwargs: Any) -> Case:
    kwargs.setdefault("id", "c1")
    kwargs.setdefault("input", "question")
    return Case(**kwargs)
