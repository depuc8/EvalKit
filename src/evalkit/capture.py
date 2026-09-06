"""Framework-neutral trace capture.

:class:`TraceRecorder` is the core: adapters (e.g. the LangGraph callback) feed it
step events, and it assembles a :class:`~evalkit.trace.Trace`. For plain Python
agents, use the :func:`traced` decorator or the :func:`trace_run` context manager.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from evalkit.trace import LLMCall, Step, StepKind, TokenUsage, ToolCall, Trace, TraceStore

F = TypeVar("F", bound=Callable[..., Any])

TraceExporter = Callable[[Trace], None]
StepListener = Callable[[Step], None]

_current_recorder: ContextVar[TraceRecorder | None] = ContextVar(
    "evalkit_current_recorder", default=None
)


def to_jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-compatible values."""
    if _depth > 12:
        return repr(obj)
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump(mode="python"), _depth=_depth + 1)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), _depth=_depth + 1)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth=_depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v, _depth=_depth + 1) for v in obj]
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return repr(obj)


def current_recorder() -> TraceRecorder | None:
    """The recorder of the run currently executing in this context, if any."""
    return _current_recorder.get()


@contextmanager
def use_recorder(recorder: TraceRecorder) -> Iterator[TraceRecorder]:
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)


class TraceRecorder:
    """Thread-safe builder for a single :class:`Trace`."""

    def __init__(
        self,
        *,
        agent: str = "agent",
        input: Any = None,
        case_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        context_tools: Iterable[str] = (),
        on_step: StepListener | None = None,
    ) -> None:
        self.context_tools = frozenset(context_tools)
        self._on_step = on_step
        self._lock = threading.Lock()
        self._perf_start = time.perf_counter()
        self._open: dict[str, tuple[Step, float]] = {}
        self._first_token: dict[str, float] = {}
        self._finished: Trace | None = None
        self.result: Any = None
        self.trace = Trace(
            agent=agent,
            input=to_jsonable(input),
            case_id=case_id,
            metadata=dict(metadata or {}),
        )
        if trace_id:
            self.trace.trace_id = trace_id

    # -- low-level step API -------------------------------------------------
    def begin_step(
        self,
        name: str,
        kind: StepKind = "custom",
        *,
        inputs: Any = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        step = Step(
            name=name,
            kind=kind,
            parent_id=parent_id,
            inputs=to_jsonable(inputs),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._open[step.id] = (step, time.perf_counter())
        return step.id

    def mark_first_token(self, step_id: str) -> None:
        with self._lock:
            entry = self._open.get(step_id)
            if entry is not None and step_id not in self._first_token:
                self._first_token[step_id] = time.perf_counter() - entry[1]

    def end_step(
        self,
        step_id: str,
        *,
        outputs: Any = None,
        error: str | None = None,
        llm: LLMCall | None = None,
        tool: ToolCall | None = None,
        documents: Sequence[str] | None = None,
    ) -> Step | None:
        with self._lock:
            entry = self._open.pop(step_id, None)
            ttft = self._first_token.pop(step_id, None)
            if self._finished is not None:
                return None  # late event from an abandoned (timed-out) run
        if entry is None:
            return None
        step, started = entry
        step.latency_s = time.perf_counter() - started
        step.outputs = to_jsonable(outputs)
        step.error = error
        if llm is not None:
            if ttft is not None and llm.time_to_first_token_s is None:
                llm.time_to_first_token_s = ttft
            step.llm = llm
        if tool is not None:
            step.tool = tool
            if tool.name in self.context_tools and tool.error is None and not documents:
                documents = normalize_documents(tool.result)
        if documents:
            step.documents = list(documents)
        with self._lock:
            self.trace.steps.append(step)
        if self._on_step is not None:
            self._on_step(step)
        return step

    # -- convenience API for hand-instrumented agents ------------------------
    @contextmanager
    def step(
        self,
        name: str,
        kind: StepKind = "custom",
        *,
        inputs: Any = None,
        parent_id: str | None = None,
    ) -> Iterator[StepHandle]:
        """Record a step around a ``with`` block; exceptions are recorded and re-raised."""
        step_id = self.begin_step(name, kind, inputs=inputs, parent_id=parent_id)
        handle = StepHandle(step_id)
        try:
            yield handle
        except BaseException as exc:
            self.end_step(
                step_id,
                outputs=handle.outputs,
                error=f"{type(exc).__name__}: {exc}",
                llm=handle.llm,
                tool=handle.tool,
                documents=handle.documents,
            )
            raise
        self.end_step(
            step_id,
            outputs=handle.outputs,
            llm=handle.llm,
            tool=handle.tool,
            documents=handle.documents,
        )

    def record_llm(
        self,
        *,
        model: str | None,
        prompt: str | Sequence[dict[str, Any]],
        output: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_s: float = 0.0,
        time_to_first_token_s: float | None = None,
        name: str | None = None,
        cached: bool = False,
        tool_calls: Sequence[dict[str, Any]] = (),
        parent_id: str | None = None,
    ) -> Step:
        """Record an already-completed LLM call."""
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        )
        step_id = self.begin_step(
            name or model or "llm", "llm", inputs=messages, parent_id=parent_id
        )
        with self._lock:
            step, started = self._open[step_id]
            self._open[step_id] = (step, started - latency_s)
        call = LLMCall(
            model=model,
            prompt=messages,
            output=output,
            tool_calls=[dict(c) for c in tool_calls],
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            time_to_first_token_s=time_to_first_token_s,
            cached=cached,
        )
        done = self.end_step(step_id, outputs=output, llm=call)
        assert done is not None
        return done

    def record_tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        result: Any = None,
        error: str | None = None,
        latency_s: float = 0.0,
        parent_id: str | None = None,
    ) -> Step:
        """Record an already-completed tool call."""
        step_id = self.begin_step(name, "tool", inputs=args, parent_id=parent_id)
        with self._lock:
            step, started = self._open[step_id]
            self._open[step_id] = (step, started - latency_s)
        call = ToolCall(
            name=name, args=to_jsonable(args or {}), result=to_jsonable(result), error=error
        )
        done = self.end_step(step_id, outputs=result, error=error, tool=call)
        assert done is not None
        return done

    # -- completion -----------------------------------------------------------
    def finish(self, output: Any = None, error: str | None = None) -> Trace:
        """Finalise and return the trace. Idempotent."""
        if self._finished is not None:
            return self._finished
        with self._lock:
            leftovers = list(self._open.values())
            self._open.clear()
        for step, started in leftovers:
            step.latency_s = time.perf_counter() - started
            step.error = step.error or "step never completed"
            self.trace.steps.append(step)
        self.trace.latency_s = time.perf_counter() - self._perf_start
        self.trace.output = to_jsonable(output)
        self.trace.error = error
        self.trace.steps.sort(key=lambda s: s.start_time)
        self._finished = self.trace
        return self.trace


class StepHandle:
    """Mutable handle yielded by :meth:`TraceRecorder.step`."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        self.outputs: Any = None
        self.llm: LLMCall | None = None
        self.tool: ToolCall | None = None
        self.documents: list[str] | None = None


def normalize_documents(value: Any) -> list[str]:
    """Turn retriever/tool output into a list of document strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("page_content", "content", "text"):
            if isinstance(value.get(key), str):
                return [value[key]]
        return [str(to_jsonable(value))]
    if isinstance(value, list | tuple):
        docs: list[str] = []
        for item in value:
            docs.extend(normalize_documents(item))
        return docs
    page_content = getattr(value, "page_content", None)
    if isinstance(page_content, str):
        return [page_content]
    return [str(value)]


@contextmanager
def trace_run(
    *,
    agent: str = "agent",
    input: Any = None,
    store: TraceStore | str | Path | None = None,
    exporters: Sequence[TraceExporter] = (),
    metadata: dict[str, Any] | None = None,
    case_id: str | None = None,
    context_tools: Iterable[str] = (),
) -> Iterator[TraceRecorder]:
    """Record everything that happens inside the ``with`` block as one trace.

    Set the final answer with ``recorder.result = ...`` (or let :func:`traced` do it).
    """
    recorder = TraceRecorder(
        agent=agent,
        input=input,
        case_id=case_id,
        metadata=metadata,
        context_tools=context_tools,
    )
    error: str | None = None
    try:
        with use_recorder(recorder):
            yield recorder
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace = recorder.finish(output=recorder.result, error=error)
        _emit(trace, store, exporters)


def _emit(
    trace: Trace, store: TraceStore | str | Path | None, exporters: Sequence[TraceExporter]
) -> None:
    if store is not None:
        (store if isinstance(store, TraceStore) else TraceStore(store)).append(trace)
    for exporter in exporters:
        exporter(trace)


def _bound_input(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    values = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
    if len(values) == 1:
        return next(iter(values.values()))
    return values


def traced(
    func: F | None = None,
    *,
    agent: str | None = None,
    store: TraceStore | str | Path | None = None,
    exporters: Sequence[TraceExporter] = (),
    context_tools: Iterable[str] = (),
) -> Any:
    """Decorator: record each call of a (sync or async) agent function as a trace.

    Inside the function, use :func:`current_recorder` to add LLM/tool steps::

        @traced(store="traces.jsonl")
        def my_agent(question: str) -> str:
            rec = current_recorder()
            ...
    """

    def decorate(fn: F) -> F:
        name = agent or fn.__name__

        def make_ctx(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            return trace_run(
                agent=name,
                input=_bound_input(fn, args, kwargs),
                store=store,
                exporters=exporters,
                context_tools=context_tools,
            )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with make_ctx(args, kwargs) as rec:
                    result = await fn(*args, **kwargs)
                    rec.result = result
                    return result

            return cast("F", awrapper)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with make_ctx(args, kwargs) as rec:
                result = fn(*args, **kwargs)
                rec.result = result
                return result

        return cast("F", wrapper)

    return decorate(func) if func is not None else decorate
