from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from evalkit.adapters import (
    AgentAdapter,
    AgentLoadError,
    FunctionAdapter,
    adapt,
    register_adapter,
    resolve_agent,
)
from evalkit.adapters import base as adapters_base
from evalkit.capture import TraceRecorder, current_recorder
from evalkit.dataset import Case
from evalkit.registry import ScorerSpecError, build_scorers
from evalkit.runner import EvalResults, run_case, run_eval, run_eval_async, score_case
from evalkit.scorers import AccuracyScorer, LatencyCostScorer, Score, Scorer
from evalkit.trace import TraceStore
from tests.helpers import make_case


def echo(question: Any) -> str:
    rec = current_recorder()
    assert rec is not None
    rec.record_tool("lookup", {"q": question}, result="data")
    return f"echo: {question}"


CASES = [
    Case(id=f"c{i}", input=f"q{i}", reference=f"echo: q{i}" if i % 2 == 0 else "mismatch")
    for i in range(6)
]


def test_run_eval_scores_and_summarises(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.jsonl")
    seen: list[int] = []
    results = run_eval(
        CASES,
        FunctionAdapter(echo, name="echoer"),
        [AccuracyScorer("exact"), LatencyCostScorer()],
        concurrency=3,
        store=store,
        on_case=lambda done, total, r: seen.append(done),
    )
    assert results.agent == "echoer" and len(results.cases) == 6
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
    summary = results.summary()
    assert summary["accuracy"].mean == pytest.approx(0.5)
    assert summary["accuracy"].pass_rate == pytest.approx(0.5)
    assert results.mean("accuracy") == pytest.approx(0.5) and results.mean("nope") is None
    assert results.pass_rate == pytest.approx(0.5) and len(results.failures) == 3
    assert "latency_p95_s" in results.metrics["latency_cost"]
    assert summary["latency_cost"].metrics == results.metrics["latency_cost"]
    assert len(store.read_all()) == 6 and len(results.traces) == 6
    first = results.cases[0]
    assert first.score_for("accuracy") is not None and first.score_for("zzz") is None
    assert first.trace.tool_calls[0].name == "lookup"


def test_run_eval_runs_concurrently() -> None:
    active, peak = 0, 0
    lock = threading.Lock()

    def slow(question: Any) -> str:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "x"

    run_eval(CASES, FunctionAdapter(slow), [], concurrency=4)
    assert 2 <= peak <= 4


def test_run_eval_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        run_eval(CASES, FunctionAdapter(echo), [], concurrency=0)


def test_agent_errors_become_failed_cases() -> None:
    def broken(question: Any) -> str:
        raise RuntimeError("model unavailable")

    results = run_eval(CASES[:2], FunctionAdapter(broken), [AccuracyScorer("exact")])
    assert all(c.error == "RuntimeError: model unavailable" for c in results.cases)
    assert not any(c.passed for c in results.cases)


def test_timeout_marks_case_failed() -> None:
    def hang(question: Any) -> str:
        time.sleep(0.5)
        return "late"

    results = run_eval(CASES[:1], FunctionAdapter(hang), [], timeout_s=0.05)
    assert results.cases[0].error is not None and "TimeoutError" in results.cases[0].error
    assert not results.cases[0].passed


def test_scorer_crash_is_contained() -> None:
    class Crashy(Scorer):
        default_name = "crashy"

        def score(self, case: Case, trace: Any) -> Score:
            raise ValueError("oops")

    (score,) = score_case(make_case(), TraceRecorder().finish(), [Crashy()])
    assert not score.passed and score.score == 0.0 and score.error == "ValueError: oops"


def test_exporters_receive_traces() -> None:
    exported: list[str] = []
    run_eval(
        CASES[:2], FunctionAdapter(echo), [], exporters=[lambda t: exported.append(t.trace_id)]
    )
    assert len(exported) == 2


async def test_run_eval_inside_running_loop_and_async_agent() -> None:
    async def agent(question: Any) -> str:
        await asyncio.sleep(0)
        return "async"

    results = await run_eval_async(CASES[:2], FunctionAdapter(agent), [])
    assert [c.trace.answer for c in results.cases] == ["async", "async"]
    # the sync wrapper must also work when a loop is already running (e.g. notebooks)
    nested = run_eval(CASES[:2], FunctionAdapter(agent), [])
    assert len(nested.cases) == 2


def test_sync_invoke_of_async_function_adapter() -> None:
    async def agent(question: Any) -> str:
        return "ok"

    assert FunctionAdapter(agent).invoke("q", TraceRecorder()) == "ok"


async def test_run_case_direct() -> None:
    trace = await run_case(CASES[0], FunctionAdapter(echo))
    assert trace.case_id == "c0" and trace.output == "echo: q0"


def test_empty_results_are_safe() -> None:
    empty = EvalResults()
    assert empty.pass_rate == 0.0 and empty.summary() == {} and empty.failures == []


# ---- adapters --------------------------------------------------------------------------


def test_adapt_dispatch() -> None:
    class Mine(AgentAdapter):
        name = "mine"

        def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
            return input

    mine = Mine()
    assert adapt(mine) is mine
    assert isinstance(adapt(echo), FunctionAdapter)
    assert mine.context_tools == frozenset()
    with pytest.raises(AgentLoadError, match="don't know how to run"):
        adapt(42)


async def test_default_ainvoke_uses_thread() -> None:
    class Sync(AgentAdapter):
        def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
            return threading.current_thread().name

    name = await Sync().ainvoke("q", TraceRecorder())
    assert name != threading.current_thread().name


def test_register_adapter_takes_priority() -> None:
    class Widget:
        pass

    class WidgetAdapter(AgentAdapter):
        def invoke(self, input: Any, recorder: TraceRecorder) -> Any:
            return "widget"

    snapshot = list(adapters_base._REGISTRY)
    try:
        register_adapter(lambda o: isinstance(o, Widget), lambda o: WidgetAdapter())
        assert isinstance(adapt(Widget()), WidgetAdapter)
    finally:
        adapters_base._REGISTRY[:] = snapshot


def test_function_adapter_context_tools() -> None:
    adapter = FunctionAdapter(echo, context_tools=frozenset({"lookup"}))
    assert adapter.context_tools == {"lookup"} and adapter.name == "echo"


def test_resolve_agent_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "agentpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def run(q):\n    return q\nvalue = 3\n")
    (tmp_path / "solo.py").write_text("def run(q):\n    return 'solo'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(resolve_agent("agentpkg:run"), FunctionAdapter)
    assert resolve_agent("solo.py:run").invoke("x", TraceRecorder()) == "solo"
    for bad, message in [
        ("nocolon", "invalid agent spec"),
        (":attr", "invalid agent spec"),
        ("missing_mod_xyz:run", "cannot import"),
        ("agentpkg:nope", "no attribute"),
        ("nofile.py:run", "not found"),
        ("agentpkg:value", "don't know how to run"),
    ]:
        with pytest.raises(AgentLoadError, match=message):
            resolve_agent(bad)


# ---- registry ----------------------------------------------------------------------------


class _Custom(Scorer):
    default_name = "custom"

    def score(self, case: Case, trace: Any) -> Score:
        return self.result(1.0, "fine")


def test_build_scorers() -> None:
    names = [s.name for s in build_scorers()]
    assert names == ["accuracy", "groundedness", "safety", "latency_cost", "tool_use"]
    custom = build_scorers(["tests.test_runner_and_adapters:_Custom"])
    assert custom[0].name == "custom"
    instance = build_scorers(["tests.test_runner_and_adapters:INSTANCE"])
    assert instance[0] is INSTANCE
    judged = build_scorers(["accuracy"], judge=lambda p: "{}", accuracy_mode="judge")
    assert isinstance(judged[0], AccuracyScorer) and judged[0].mode == "judge"
    for bad in (
        "bogus",
        "nomodule_xyz:Thing",
        "tests.test_runner_and_adapters:missing",
        "json:dumps",
    ):
        with pytest.raises(ScorerSpecError):
            build_scorers([bad])


INSTANCE = _Custom()
