"""LLM-as-judge plumbing.

A *judge* is any callable ``(prompt: str) -> str``. Scorers build the prompt and parse
the reply as JSON, so any provider can be plugged in.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from typing import Any

Judge = Callable[[str], str]


class JudgeError(RuntimeError):
    """The judge failed or returned something we could not parse."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from ``text`` (tolerates fences and chatter)."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise JudgeError(f"judge did not return a JSON object: {text[:200]!r}")


def ask_json(judge: Judge, prompt: str) -> dict[str, Any]:
    """Call the judge and parse its reply, wrapping any failure in :class:`JudgeError`."""
    try:
        reply = judge(prompt)
    except JudgeError:
        raise
    except Exception as exc:
        raise JudgeError(f"judge call failed: {type(exc).__name__}: {exc}") from exc
    return parse_json_object(reply)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


class AnthropicJudge:
    """Judge backed by the Anthropic Messages API (``pip install evalkit-agents[anthropic]``)."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        client: Any = None,
        max_tokens: int = 1024,
        system: str = "You are a strict evaluation judge. Reply with a single JSON object only.",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise JudgeError(
                    "the anthropic package is required: pip install 'evalkit-agents[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self._client = client

    def __call__(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in message.content)


def load_judge(spec: str) -> Judge:
    """Build a judge from a CLI spec: ``anthropic[:model]`` or ``module:callable``."""
    if spec == "anthropic" or spec.startswith("anthropic:"):
        _, _, model = spec.partition(":")
        return AnthropicJudge(model) if model else AnthropicJudge()
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise JudgeError(f"invalid judge spec {spec!r}; use 'anthropic[:model]' or 'module:attr'")
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise JudgeError(f"cannot load judge {spec!r}: {exc}") from exc
    if not callable(obj):
        raise JudgeError(f"judge {spec!r} is not callable")
    judge: Judge = obj
    return judge
