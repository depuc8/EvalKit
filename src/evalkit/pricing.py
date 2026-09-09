"""Token pricing configuration used by the latency & cost scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from evalkit.trace import TokenUsage, Trace

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class ModelPrice(BaseModel):
    """USD price per one million tokens."""

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)

    def cost(self, usage: TokenUsage) -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


class PricingConfig(BaseModel):
    """Maps model names to prices. Lookup is exact, then longest-prefix, then ``default``."""

    models: dict[str, ModelPrice] = Field(default_factory=dict)
    default: ModelPrice | None = None

    @classmethod
    def load(cls, path: str | Path) -> PricingConfig:
        """Load from a ``.json`` or ``.toml`` file."""
        p = Path(path)
        if p.suffix == ".toml":
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def price_for(self, model: str | None) -> ModelPrice | None:
        if model:
            if model in self.models:
                return self.models[model]
            prefixes = [k for k in self.models if model.startswith(k)]
            if prefixes:
                return self.models[max(prefixes, key=len)]
        return self.default

    def trace_cost(self, trace: Trace) -> float | None:
        """Total USD cost of a trace, or ``None`` if it made no priced LLM calls."""
        total = 0.0
        priced = False
        for call in trace.llm_calls:
            price = self.price_for(call.model)
            if price is not None:
                total += price.cost(call.usage)
                priced = True
        return total if priced else None

    def unpriced_models(self, trace: Trace) -> list[str]:
        return sorted(
            {c.model or "<unknown>" for c in trace.llm_calls if self.price_for(c.model) is None}
        )
