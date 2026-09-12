"""Latency & cost: p50/p95 latency, time-to-first-token, tokens/sec and cost per request."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from evalkit.dataset import Case
from evalkit.pricing import PricingConfig
from evalkit.scorers.base import Score, ScoredRow, Scorer
from evalkit.trace import Trace


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]. Raises on empty input."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def time_to_first_token(trace: Trace) -> float | None:
    """TTFT of the first streamed LLM call, if any call was streamed."""
    for call in trace.llm_calls:
        if call.time_to_first_token_s is not None:
            return call.time_to_first_token_s
    return None


def tokens_per_second(trace: Trace) -> float | None:
    """Output tokens divided by generation time across all (non-cached) LLM calls."""
    tokens, seconds = 0, 0.0
    for step in trace.llm_steps:
        call = step.llm
        assert call is not None
        if call.cached or call.usage.output_tokens == 0:
            continue
        tokens += call.usage.output_tokens
        gen = step.latency_s - (call.time_to_first_token_s or 0.0)
        seconds += max(gen, 1e-9)
    return tokens / seconds if tokens and seconds > 0 else None


class LatencyCostScorer(Scorer):
    """Measures speed and spend; optionally enforces per-request budgets.

    Without budgets the score is always 1.0 and the numbers are informational. With
    ``max_latency_s`` / ``max_cost_usd`` / ``max_ttft_s`` the score is the worst
    ``budget / actual`` ratio (capped at 1.0), so exceeding a budget lowers it.
    Run-level p50/p95 figures come from :meth:`aggregate`.
    """

    default_name: ClassVar[str] = "latency_cost"
    default_threshold: ClassVar[float] = 1.0

    def __init__(
        self,
        *,
        pricing: PricingConfig | None = None,
        max_latency_s: float | None = None,
        max_cost_usd: float | None = None,
        max_ttft_s: float | None = None,
        name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.pricing = pricing
        self.max_latency_s = max_latency_s
        self.max_cost_usd = max_cost_usd
        self.max_ttft_s = max_ttft_s

    def measure(self, trace: Trace) -> dict[str, float | None]:
        usage = trace.usage
        return {
            "latency_s": trace.latency_s,
            "ttft_s": time_to_first_token(trace),
            "tokens_per_second": tokens_per_second(trace),
            "input_tokens": float(usage.input_tokens),
            "output_tokens": float(usage.output_tokens),
            "cost_usd": self.pricing.trace_cost(trace) if self.pricing else None,
        }

    def score(self, case: Case, trace: Trace) -> Score:
        m = self.measure(trace)
        ratios: list[float] = []
        notes = [f"latency {trace.latency_s:.3f}s"]
        if m["ttft_s"] is not None:
            notes.append(f"ttft {m['ttft_s']:.3f}s")
        if m["tokens_per_second"] is not None:
            notes.append(f"{m['tokens_per_second']:.0f} tok/s")
        if m["cost_usd"] is not None:
            notes.append(f"cost ${m['cost_usd']:.6f}")
        for label, actual, budget in (
            ("latency", m["latency_s"], self.max_latency_s),
            ("ttft", m["ttft_s"], self.max_ttft_s),
            ("cost", m["cost_usd"], self.max_cost_usd),
        ):
            if budget is None or actual is None:
                continue
            ratios.append(1.0 if actual <= budget else budget / actual)
            if actual > budget:
                notes.append(f"{label} over budget ({actual:.4g} > {budget:.4g})")
        unpriced = self.pricing.unpriced_models(trace) if self.pricing else []
        if unpriced:
            notes.append("no pricing for " + ", ".join(unpriced))
        return self.result(min(ratios, default=1.0), ", ".join(notes), **m)

    def aggregate(self, rows: Sequence[ScoredRow]) -> dict[str, float]:
        traces = [r.trace for r in rows]
        if not traces:
            return {}
        out: dict[str, float] = {}
        lat = [t.latency_s for t in traces]
        out["latency_p50_s"] = percentile(lat, 50)
        out["latency_p95_s"] = percentile(lat, 95)
        ttfts = [v for t in traces if (v := time_to_first_token(t)) is not None]
        if ttfts:
            out["ttft_p50_s"] = percentile(ttfts, 50)
            out["ttft_p95_s"] = percentile(ttfts, 95)
        tps = [v for t in traces if (v := tokens_per_second(t)) is not None]
        if tps:
            out["tokens_per_second_mean"] = sum(tps) / len(tps)
        out["input_tokens_mean"] = sum(t.usage.input_tokens for t in traces) / len(traces)
        out["output_tokens_mean"] = sum(t.usage.output_tokens for t in traces) / len(traces)
        if self.pricing:
            costs = [c for t in traces if (c := self.pricing.trace_cost(t)) is not None]
            if costs:
                out["cost_per_request_usd"] = sum(costs) / len(costs)
                out["total_cost_usd"] = sum(costs)
        return out
