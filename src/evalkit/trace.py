"""Structured trace data model and JSON-lines storage."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["node", "llm", "tool", "retriever", "custom"]


def new_id() -> str:
    """Return a short unique id (32 hex chars)."""
    return uuid.uuid4().hex


class TokenUsage(BaseModel):
    """Token counts for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class LLMCall(BaseModel):
    """A single model invocation: the prompt that went in and what came out."""

    model: str | None = None
    prompt: list[dict[str, Any]] = Field(default_factory=list)
    output: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    time_to_first_token_s: float | None = None
    cached: bool = False

    @property
    def prompt_text(self) -> str:
        return "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in self.prompt)


class ToolCall(BaseModel):
    """A tool invocation with its arguments and result."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class Step(BaseModel):
    """One unit of work inside a run (graph node, LLM call, tool call, retrieval)."""

    id: str = Field(default_factory=new_id)
    name: str
    kind: StepKind = "custom"
    parent_id: str | None = None
    start_time: float = Field(default_factory=time.time)
    latency_s: float = 0.0
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    llm: LLMCall | None = None
    tool: ToolCall | None = None
    documents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """A complete agent run."""

    trace_id: str = Field(default_factory=new_id)
    agent: str = "agent"
    case_id: str | None = None
    start_time: float = Field(default_factory=time.time)
    latency_s: float = 0.0
    input: Any = None
    output: Any = None
    error: str | None = None
    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def llm_steps(self) -> list[Step]:
        return [s for s in self.steps if s.llm is not None]

    @property
    def llm_calls(self) -> list[LLMCall]:
        return [s.llm for s in self.steps if s.llm is not None]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool for s in self.steps if s.tool is not None]

    @property
    def usage(self) -> TokenUsage:
        total = TokenUsage()
        for call in self.llm_calls:
            total = total + call.usage
        return total

    @property
    def answer(self) -> str:
        """The final output rendered as text ('' when the run produced none)."""
        if self.output is None:
            return ""
        if isinstance(self.output, str):
            return self.output
        return str(self.output)

    def retrieved_context(self) -> list[str]:
        """All documents surfaced by retriever steps or declared context tools."""
        return [doc for step in self.steps for doc in step.documents]


class TraceStore:
    """Append-only JSON-lines trace file. Safe to share between threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, trace: Trace) -> None:
        line = trace.model_dump_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def __iter__(self) -> Iterator[Trace]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield Trace.model_validate_json(line)

    def read_all(self) -> list[Trace]:
        return list(self)

    def get(self, trace_id: str) -> Trace:
        """Look up a trace by full id or unique prefix."""
        matches = [t for t in self if t.trace_id.startswith(trace_id)]
        if not matches:
            raise KeyError(f"no trace matching {trace_id!r} in {self.path}")
        if len(matches) > 1:
            raise KeyError(f"trace id prefix {trace_id!r} is ambiguous ({len(matches)} matches)")
        return matches[0]
