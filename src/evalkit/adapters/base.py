"""The adapter interface: how EvalKit runs an agent built with any framework."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evalkit.capture import TraceRecorder


class AgentLoadError(RuntimeError):
    """An ``--agent module:attr`` spec could not be resolved."""


class AgentAdapter(ABC):
    """Runs one case through an agent while a :class:`TraceRecorder` captures what happens.

    To support a new framework, subclass this, record steps on ``recorder`` (for example by
    registering a framework callback that calls ``recorder.begin_step``/``end_step``) and
    return the agent's final output. Then register it with :func:`register_adapter`.
    """

    name: str = "agent"

    @abstractmethod
    def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
        """Run the agent synchronously and return its final output."""

    async def ainvoke(self, input: Any, recorder: TraceRecorder) -> Any:
        """Async entry point. Defaults to running :meth:`invoke` in a worker thread."""
        return await asyncio.to_thread(self.invoke, input, recorder)

    @property
    def context_tools(self) -> frozenset[str]:
        """Tool names whose results count as retrieved context (for groundedness)."""
        return frozenset()


class FunctionAdapter(AgentAdapter):
    """Wraps a plain ``fn(input) -> output`` (sync or async).

    Inside the function use :func:`evalkit.capture.current_recorder` to log LLM and tool steps.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        name: str | None = None,
        context_tools: frozenset[str] = frozenset(),
    ) -> None:
        self.fn = fn
        self.name = name or str(getattr(fn, "__name__", "function"))
        self._context_tools = context_tools

    @property
    def context_tools(self) -> frozenset[str]:
        return self._context_tools

    def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
        result = self.fn(input)
        if inspect.isawaitable(result):
            return asyncio.run(_await(result))
        return result

    async def ainvoke(self, input: Any, recorder: TraceRecorder) -> Any:
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(input)
        return await asyncio.to_thread(self.invoke, input, recorder)


async def _await(awaitable: Any) -> Any:
    return await awaitable


# -- registry ---------------------------------------------------------------------------
Predicate = Callable[[Any], bool]
Factory = Callable[[Any], AgentAdapter]

_REGISTRY: list[tuple[Predicate, Factory]] = []


def register_adapter(predicate: Predicate, factory: Factory) -> None:
    """Teach :func:`resolve_agent` to wrap objects for which ``predicate(obj)`` is true."""
    _REGISTRY.insert(0, (predicate, factory))


def _is_langgraph(obj: Any) -> bool:
    return type(obj).__module__.startswith("langgraph") and hasattr(obj, "invoke")


def _langgraph_factory(obj: Any) -> AgentAdapter:
    from evalkit.adapters.langgraph import LangGraphAdapter

    return LangGraphAdapter(obj)


register_adapter(_is_langgraph, _langgraph_factory)


def adapt(obj: Any) -> AgentAdapter:
    """Turn a framework object (or plain callable) into an :class:`AgentAdapter`."""
    if isinstance(obj, AgentAdapter):
        return obj
    for predicate, factory in _REGISTRY:
        if predicate(obj):
            return factory(obj)
    if callable(obj):
        return FunctionAdapter(obj)
    raise AgentLoadError(
        f"don't know how to run {type(obj).__name__!r}; wrap it in an AgentAdapter"
    )


def _import_target(module_spec: str) -> Any:
    path = Path(module_spec)
    if module_spec.endswith(".py") or path.is_file():
        if not path.is_file():
            raise AgentLoadError(f"agent file not found: {module_spec}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise AgentLoadError(f"cannot import {module_spec}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        sys.path.insert(0, str(path.resolve().parent))
        spec.loader.exec_module(module)
        return module
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(module_spec)


def resolve_agent(spec: str) -> AgentAdapter:
    """Load ``module:attr`` or ``path/to/file.py:attr`` and wrap it in an adapter.

    ``attr`` may be an :class:`AgentAdapter`, a framework object such as a compiled
    LangGraph graph, or a plain callable. A zero-argument factory returning one of those is
    not guessed at; export the object itself.
    """
    module_spec, sep, attr = spec.rpartition(":")
    if not sep or not module_spec or not attr:
        raise AgentLoadError(f"invalid agent spec {spec!r}; expected 'module:attr'")
    try:
        module = _import_target(module_spec)
    except ImportError as exc:
        raise AgentLoadError(f"cannot import {module_spec!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise AgentLoadError(f"{module_spec!r} has no attribute {attr!r}") from exc
    return adapt(obj)
