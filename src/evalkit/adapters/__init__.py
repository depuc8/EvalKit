"""Framework adapters."""

from __future__ import annotations

from evalkit.adapters.base import (
    AgentAdapter,
    AgentLoadError,
    FunctionAdapter,
    adapt,
    register_adapter,
    resolve_agent,
)

__all__ = [
    "AgentAdapter",
    "AgentLoadError",
    "FunctionAdapter",
    "adapt",
    "register_adapter",
    "resolve_agent",
]
