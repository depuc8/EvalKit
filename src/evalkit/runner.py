"""Run a dataset through an agent, score every case, and collect results."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from evalkit.adapters.base import AgentAdapter
from evalkit.capture import TraceExporter, TraceRecorder, use_recorder
from evalkit.dataset import Case
from evalkit.scorers.base import Score, ScoredRow, Scorer
from evalkit.trace import Trace, TraceStore, new_id


class CaseResult(BaseModel):
    """A case, the trace of its run, and its scores."""

    case: Case
    trace: Trace
    scores: list[Score] = Field(default_factory=list)

    @property
    def error(self) -> str | None:
        return self.trace.error

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores if not s.skipped)

    def score_for(self, scorer: str) -> Score | None:
        return next((s for s in self.scores if s.scorer == scorer), None)


class ScorerSummary(BaseModel):
    """Run-level statistics for one scorer."""

    mean: float | None
    pass_rate: float | None
    n: int
    skipped: int
    metrics: dict[str, float] = Field(default_factory=dict)


class EvalResults(BaseModel):
    """Everything produced by a run."""

    run_id: str = Field(default_factory=new_id)
    agent: str = "agent"
    started_at: float = Field(default_factory=time.time)
    duration_s: float = 0.0
    cases: list[CaseResult] = Field(default_factory=list)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)

    @property
    def traces(self) -> list[Trace]:
        return [c.trace for c in self.cases]

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    def scorer_names(self) -> list[str]:
        names: list[str] = []
        for c in self.cases:
            for s in c.scores:
                if s.scorer not in names:
                    names.append(s.scorer)
        return names

    def summary(self) -> dict[str, ScorerSummary]:
        out: dict[str, ScorerSummary] = {}
        for name in self.scorer_names():
            scores = [s for c in self.cases for s in c.scores if s.scorer == name]
            live = [s for s in scores if not s.skipped]
            out[name] = ScorerSummary(
                mean=sum(s.score for s in live) / len(live) if live else None,
                pass_rate=sum(s.passed for s in live) / len(live) if live else None,
                n=len(live),
                skipped=len(scores) - len(live),
                metrics=self.metrics.get(name, {}),
            )
        return out

    def mean(self, scorer: str) -> float | None:
        item = self.summary().get(scorer)
        return item.mean if item else None


def score_case(case: Case, trace: Trace, scorers: Sequence[Scorer]) -> list[Score]:
    """Run every scorer on one trace; a crashing scorer yields a failed score, not an exception."""
    scores: list[Score] = []
    for scorer in scorers:
        try:
            scores.append(scorer.score(case, trace))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            scores.append(
                Score(
                    scorer=scorer.name,
                    score=0.0,
                    passed=False,
                    reason=f"scorer raised {message}",
                    error=message,
                )
            )
    return scores


async def run_case(
    case: Case,
    adapter: AgentAdapter,
    *,
    timeout_s: float | None = None,
    exporters: Sequence[TraceExporter] = (),
    store: TraceStore | None = None,
) -> Trace:
    """Run one case and return its (finished, stored, exported) trace."""
    recorder = TraceRecorder(
        agent=adapter.name,
        input=case.input,
        case_id=case.id,
        context_tools=adapter.context_tools,
    )
    output: Any = None
    error: str | None = None
    try:
        with use_recorder(recorder):
            coro = adapter.ainvoke(case.input, recorder)
            output = await (asyncio.wait_for(coro, timeout_s) if timeout_s else coro)
    except asyncio.TimeoutError:
        error = f"TimeoutError: no result after {timeout_s}s"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    trace = recorder.finish(output=output, error=error)
    if store is not None:
        store.append(trace)
    for exporter in exporters:
        exporter(trace)
    return trace


async def run_eval_async(
    cases: Sequence[Case],
    adapter: AgentAdapter,
    scorers: Sequence[Scorer],
    *,
    concurrency: int = 4,
    timeout_s: float | None = None,
    store: TraceStore | None = None,
    exporters: Sequence[TraceExporter] = (),
    on_case: Callable[[int, int, CaseResult], None] | None = None,
) -> EvalResults:
    """Run all cases with bounded concurrency, then score them."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    started = time.perf_counter()
    results = EvalResults(agent=adapter.name)
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def one(case: Case) -> CaseResult:
        nonlocal done
        async with gate:
            trace = await run_case(
                case, adapter, timeout_s=timeout_s, exporters=exporters, store=store
            )
            scores = await asyncio.to_thread(score_case, case, trace, scorers)
        result = CaseResult(case=case, trace=trace, scores=scores)
        done += 1
        if on_case is not None:
            on_case(done, len(cases), result)
        return result

    results.cases = list(await asyncio.gather(*(one(c) for c in cases)))
    for scorer in scorers:
        rows = [
            ScoredRow(c.case, c.trace, s)
            for c in results.cases
            for s in c.scores
            if s.scorer == scorer.name and not s.skipped
        ]
        metrics = scorer.aggregate(rows) if rows else {}
        if metrics:
            results.metrics[scorer.name] = metrics
    results.duration_s = time.perf_counter() - started
    return results


def run_eval(
    cases: Sequence[Case],
    adapter: AgentAdapter,
    scorers: Sequence[Scorer],
    **kwargs: Any,
) -> EvalResults:
    """Synchronous wrapper around :func:`run_eval_async` (safe inside a running event loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_eval_async(cases, adapter, scorers, **kwargs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run_eval_async(cases, adapter, scorers, **kwargs)).result()
