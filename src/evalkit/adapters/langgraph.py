"""LangGraph adapter (``pip install evalkit-agents[langgraph]``).

Three pieces:

* :class:`EvalKitCallback` - a LangChain callback handler that turns graph nodes, LLM calls,
  tool calls and retrievals into steps on a :class:`~evalkit.capture.TraceRecorder`. Use it
  on its own with any LangChain/LangGraph runnable: ``graph.invoke(x, {"callbacks": [cb]})``.
* :class:`LangGraphAdapter` - runs a compiled graph for the evaluation runner.
* :func:`replayable` / :class:`ReplayableLLM` - wrap a chat model so ``evalkit replay`` can
  serve recorded outputs, and swap the prompt or model at a chosen step.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.runnables import Runnable

from evalkit.adapters.base import AgentAdapter
from evalkit.capture import TraceRecorder, current_recorder, normalize_documents, to_jsonable
from evalkit.replay import ReplayError, StepOverride, current_plan
from evalkit.trace import LLMCall, TokenUsage, ToolCall

#: Async runs only propagate LangChain's callback context into node code on Python >= 3.11.
_ASYNC_CONTEXT_PROPAGATES = sys.version_info >= (3, 11)

_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
_ANSWER_KEYS = ("answer", "output", "response", "final_answer", "result")


def _text(content: Any) -> str:
    """Flatten message content (a string or a list of content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _message_dicts(messages: Iterable[BaseMessage]) -> list[dict[str, Any]]:
    return [{"role": _ROLES.get(m.type, m.type), "content": _text(m.content)} for m in messages]


def _model_name(serialized: Any, metadata: Any, kwargs: Mapping[str, Any]) -> str | None:
    meta = metadata or {}
    if meta.get("ls_model_name"):
        return str(meta["ls_model_name"])
    params = kwargs.get("invocation_params") or {}
    for key in ("model", "model_name", "model_id"):
        if params.get(key):
            return str(params[key])
    name = (serialized or {}).get("name") or (serialized or {}).get("id", [None])[-1]
    return str(name) if name else None


class EvalKitCallback(BaseCallbackHandler):  # type: ignore[misc]
    """Feeds LangChain/LangGraph run events into a :class:`TraceRecorder`."""

    raise_error = False

    def __init__(self, recorder: TraceRecorder) -> None:
        self.recorder = recorder
        self._lock = threading.Lock()
        self._parents: dict[UUID, UUID | None] = {}
        self._steps: dict[UUID, str] = {}
        self._llm: dict[UUID, tuple[str | None, list[dict[str, Any]]]] = {}
        self._tools: dict[UUID, tuple[str, dict[str, Any]]] = {}

    # -- helpers -----------------------------------------------------------------
    def _link(self, run_id: UUID, parent_run_id: UUID | None) -> str | None:
        """Remember the run's parent and return the nearest recorded ancestor step id."""
        with self._lock:
            self._parents[run_id] = parent_run_id
            cursor = parent_run_id
            while cursor is not None:
                if cursor in self._steps:
                    return self._steps[cursor]
                cursor = self._parents.get(cursor)
        return None

    def _begin(
        self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: Any, inputs: Any
    ) -> str:
        parent = self._link(run_id, parent_run_id)
        step_id = self.recorder.begin_step(name, kind, inputs=inputs, parent_id=parent)
        with self._lock:
            self._steps[run_id] = step_id
        return step_id

    def _step(self, run_id: UUID) -> str | None:
        with self._lock:
            return self._steps.get(run_id)

    # -- chains / graph nodes ------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        node = (metadata or {}).get("langgraph_node")
        if node and name == node:
            self._begin(run_id, parent_run_id, str(node), "node", inputs)
        else:
            self._link(run_id, parent_run_id)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            self.recorder.end_step(step_id, outputs=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            self.recorder.end_step(step_id, error=f"{type(error).__name__}: {error}")

    # -- LLMs ----------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompt = _message_dicts(messages[0]) if messages else []
        self._start_llm(run_id, parent_run_id, serialized, metadata, kwargs, prompt)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompt = [{"role": "user", "content": p} for p in prompts[:1]]
        self._start_llm(run_id, parent_run_id, serialized, metadata, kwargs, prompt)

    def _start_llm(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        serialized: Any,
        metadata: Any,
        kwargs: Mapping[str, Any],
        prompt: list[dict[str, Any]],
    ) -> None:
        model = _model_name(serialized, metadata, kwargs)
        self._begin(run_id, parent_run_id, model or "llm", "llm", prompt)
        with self._lock:
            self._llm[run_id] = (model, prompt)

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            self.recorder.mark_first_token(step_id)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if not step_id:
            return
        with self._lock:
            model, prompt = self._llm.pop(run_id, (None, []))
        text, tool_calls = "", []
        usage = TokenUsage()
        generations = getattr(response, "generations", None) or [[]]
        first = generations[0][0] if generations and generations[0] else None
        if first is not None:
            text = getattr(first, "text", "") or ""
            message = getattr(first, "message", None)
            if message is not None:
                text = text or _text(message.content)
                tool_calls = [to_jsonable(tc) for tc in getattr(message, "tool_calls", []) or []]
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    usage = TokenUsage(
                        input_tokens=int(meta.get("input_tokens", 0)),
                        output_tokens=int(meta.get("output_tokens", 0)),
                    )
                resp_meta = getattr(message, "response_metadata", {}) or {}
                model = resp_meta.get("model_name") or resp_meta.get("model") or model
        if usage.total_tokens == 0:
            raw = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            usage = TokenUsage(
                input_tokens=int(raw.get("prompt_tokens", raw.get("input_tokens", 0))),
                output_tokens=int(raw.get("completion_tokens", raw.get("output_tokens", 0))),
            )
        call = LLMCall(model=model, prompt=prompt, output=text, tool_calls=tool_calls, usage=usage)
        self.recorder.end_step(step_id, outputs=text, llm=call)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            with self._lock:
                model, prompt = self._llm.pop(run_id, (None, []))
            self.recorder.end_step(
                step_id,
                error=f"{type(error).__name__}: {error}",
                llm=LLMCall(model=model, prompt=prompt),
            )

    # -- tools ---------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.get("name") or (serialized or {}).get("name") or "tool")
        args = to_jsonable(inputs) if isinstance(inputs, dict) else {"input": input_str}
        self._begin(run_id, parent_run_id, name, "tool", args)
        with self._lock:
            self._tools[run_id] = (name, args)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if not step_id:
            return
        with self._lock:
            name, args = self._tools.pop(run_id, ("tool", {}))
        result = _text(output.content) if hasattr(output, "content") else output
        self.recorder.end_step(
            step_id, outputs=result, tool=ToolCall(name=name, args=args, result=to_jsonable(result))
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if not step_id:
            return
        with self._lock:
            name, args = self._tools.pop(run_id, ("tool", {}))
        message = f"{type(error).__name__}: {error}"
        self.recorder.end_step(
            step_id, error=message, tool=ToolCall(name=name, args=args, error=message)
        )

    # -- retrievers ----------------------------------------------------------------
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.get("name") or (serialized or {}).get("name") or "retriever")
        self._begin(run_id, parent_run_id, name, "retriever", {"query": query})

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            self.recorder.end_step(
                step_id,
                outputs=[
                    {"page_content": d.page_content, "metadata": getattr(d, "metadata", {})}
                    for d in documents
                ],
                documents=normalize_documents(list(documents)),
            )

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step_id = self._step(run_id)
        if step_id:
            self.recorder.end_step(step_id, error=f"{type(error).__name__}: {error}")


class LangGraphAdapter(AgentAdapter):
    """Runs a compiled LangGraph graph (or any LangChain ``Runnable``) under trace capture.

    Args:
        graph: the compiled graph.
        input_key: state key for string case inputs (``"messages"`` wraps text as a user
            message). Dict inputs are passed through unchanged.
        output_key: state key holding the final answer; by default the first of
            ``answer/output/response/final_answer/result``, else the last message.
        context_tools: tool names whose results count as retrieved context.
        measure_ttft: stream the graph in ``messages`` mode so streaming-capable models
            report time-to-first-token. Disable if your graph misbehaves when streamed.
        config: extra ``RunnableConfig`` (e.g. ``{"recursion_limit": 50}``).

    On Python 3.10 the async entry point runs the graph through its synchronous path in a
    worker thread (see ``_ASYNC_CONTEXT_PROPAGATES``), so graphs with ``async def``-only nodes
    need Python 3.11 or newer.
    """

    def __init__(
        self,
        graph: Any,
        *,
        name: str | None = None,
        input_key: str = "question",
        output_key: str | None = None,
        context_tools: Iterable[str] = (),
        measure_ttft: bool = True,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self.name = name or getattr(graph, "name", None) or "langgraph"
        self.input_key = input_key
        self.output_key = output_key
        self._context_tools = frozenset(context_tools)
        self.measure_ttft = measure_ttft
        self.config = dict(config or {})

    @property
    def context_tools(self) -> frozenset[str]:
        return self._context_tools

    def _payload(self, case_input: Any) -> Any:
        if not isinstance(case_input, str):
            return case_input
        if self.input_key == "messages":
            return {"messages": [("user", case_input)]}
        return {self.input_key: case_input}

    def _config(self, recorder: TraceRecorder) -> dict[str, Any]:
        config = dict(self.config)
        config["callbacks"] = [*config.get("callbacks", []), EvalKitCallback(recorder)]
        return config

    def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
        payload, config = self._payload(input), self._config(recorder)
        if not self.measure_ttft:
            return self.extract_answer(self.graph.invoke(payload, config))
        state: Any = None
        for mode, data in self.graph.stream(payload, config, stream_mode=["values", "messages"]):
            if mode == "values":
                state = data
        return self.extract_answer(state)

    async def ainvoke(self, input: Any, recorder: TraceRecorder) -> Any:
        if not _ASYNC_CONTEXT_PROPAGATES:
            # Before Python 3.11 LangChain loses the callback context for calls made inside
            # nodes during an async run, so tool/LLM/retriever steps would silently go missing.
            return await asyncio.to_thread(self.invoke, input, recorder)
        payload, config = self._payload(input), self._config(recorder)
        if not self.measure_ttft:
            return self.extract_answer(await self.graph.ainvoke(payload, config))
        state: Any = None
        async for mode, data in self.graph.astream(
            payload, config, stream_mode=["values", "messages"]
        ):
            if mode == "values":
                state = data
        return self.extract_answer(state)

    def extract_answer(self, state: Any) -> Any:
        if not isinstance(state, Mapping):
            return _text(state.content) if isinstance(state, BaseMessage) else state
        if self.output_key is not None:
            return state.get(self.output_key)
        for key in _ANSWER_KEYS:
            if key in state:
                return state[key]
        messages = state.get("messages")
        if messages:
            last = messages[-1]
            return _text(last.content) if isinstance(last, BaseMessage) else last
        return dict(state)


# -- replay support --------------------------------------------------------------------
ModelFactory = Callable[[str], Any]


def _to_messages(value: Any) -> list[BaseMessage]:
    if isinstance(value, str):
        return [HumanMessage(value)]
    if hasattr(value, "to_messages"):
        return list(value.to_messages())
    return list(convert_to_messages(value))


def _apply_override(messages: list[BaseMessage], override: StepOverride) -> list[BaseMessage]:
    if override.prompt is not None:
        return [HumanMessage(override.prompt)]
    if override.system is not None:
        rest = [m for m in messages if not isinstance(m, SystemMessage)]
        return [SystemMessage(override.system), *rest]
    return messages


def _cached_message(call: LLMCall) -> AIMessage:
    return AIMessage(
        content=call.output,
        tool_calls=[
            {
                "name": tc["name"],
                "args": tc.get("args", {}),
                "id": tc.get("id"),
                "type": "tool_call",
            }
            for tc in call.tool_calls
        ],
        usage_metadata={
            "input_tokens": call.usage.input_tokens,
            "output_tokens": call.usage.output_tokens,
            "total_tokens": call.usage.total_tokens,
        },
    )


class ReplayableLLM(Runnable[Any, Any]):  # type: ignore[misc]
    """Wraps a chat model; behaves identically unless an ``evalkit replay`` is running.

    During a replay it answers early calls from the recording and applies the step override
    (new prompt / system prompt / model) to the chosen call. ``factory`` builds the model
    for a model-name override::

        llm = replayable(ChatAnthropic(model="claude-haiku-5-5"),
                         factory=lambda name: ChatAnthropic(model=name))
    """

    def __init__(
        self,
        model: Any,
        *,
        factory: ModelFactory | None = None,
        _bindings: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (),
    ) -> None:
        self._model = model
        self._factory = factory
        self._bindings = _bindings

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ReplayableLLM:
        return ReplayableLLM(
            self._model.bind_tools(tools, **kwargs),
            factory=self._factory,
            _bindings=(*self._bindings, ("bind_tools", (tools,), kwargs)),
        )

    def _swapped(self, model_name: str) -> Any:
        if self._factory is None:
            raise ReplayError(
                "swapping the model needs a factory: replayable(model, factory=lambda name: ...)"
            )
        model = self._factory(model_name)
        for method, args, kwargs in self._bindings:
            model = getattr(model, method)(*args, **kwargs)
        return model

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        plan = current_plan()
        if plan is None:
            return self._model.invoke(input, config, **kwargs)
        decision = plan.next_decision()
        if decision.cached is not None:
            recorder = current_recorder()
            if recorder is not None:
                call = decision.cached
                recorder.record_llm(
                    model=call.model,
                    prompt=call.prompt,
                    output=call.output,
                    input_tokens=call.usage.input_tokens,
                    output_tokens=call.usage.output_tokens,
                    cached=True,
                    tool_calls=call.tool_calls,
                )
            return _cached_message(decision.cached)
        model, messages = self._model, input
        if decision.override is not None:
            override = decision.override
            if override.model:
                model = self._swapped(override.model)
            messages = _apply_override(_to_messages(input), override)
        return model.invoke(messages, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)


def replayable(model: Any, *, factory: ModelFactory | None = None) -> ReplayableLLM:
    """Make ``model`` usable with ``evalkit replay`` (see :class:`ReplayableLLM`)."""
    return ReplayableLLM(model, factory=factory)
