from __future__ import annotations

import pytest

from evalkit.otel import OTelExporter
from evalkit.trace import Step, ToolCall
from tests.helpers import make_llm, make_trace


def test_pytest_plugin_exposes_assert_eval_fixture(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        from evalkit import AccuracyScorer, Case, FunctionAdapter, run_eval

        def test_quality(assert_eval):
            results = run_eval(
                [Case(id="a", input="q", reference="yes")],
                FunctionAdapter(lambda q: "yes"),
                [AccuracyScorer("exact")],
            )
            assert_eval(results, min_accuracy=0.85)

        def test_regression_is_caught(assert_eval):
            results = run_eval(
                [Case(id="a", input="q", reference="yes")],
                FunctionAdapter(lambda q: "no"),
                [AccuracyScorer("exact")],
            )
            with pytest.raises(AssertionError, match="accuracy mean 0.000"):
                assert_eval(results, min_accuracy=0.85)
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--no-cov")
    result.assert_outcomes(passed=2)


def test_otel_exporter_emits_parented_spans() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))

    trace = make_trace(
        "ok",
        llm=[make_llm("gpt-x", inp=7, out=3, ttft=0.02)],
        tools=[ToolCall(name="search", args={"q": "x"}, result="r")],
        error="agent failed",
    )
    parent = Step(name="outer", kind="node", start_time=1.0, latency_s=1.0)
    child = Step(
        name="inner",
        kind="node",
        parent_id=parent.id,
        start_time=1.1,
        latency_s=0.1,
        error="inner failed",
    )
    trace.steps = [child, parent, *trace.steps]  # child listed before its parent on purpose

    OTelExporter(provider)(trace)
    spans = {s.name: s for s in memory.get_finished_spans()}
    root = spans["agent.run agent"]
    assert root.attributes is not None and root.attributes["evalkit.trace_id"] == trace.trace_id
    assert root.status.status_code.name == "ERROR"
    assert spans["node inner"].parent.span_id == spans["node outer"].context.span_id
    assert spans["node outer"].parent.span_id == root.context.span_id
    llm = spans["llm gpt-x"]
    assert llm.attributes["gen_ai.usage.input_tokens"] == 7
    assert llm.attributes["gen_ai.usage.output_tokens"] == 3
    assert llm.attributes["evalkit.llm.time_to_first_token_s"] == 0.02
    assert "gen_ai.prompt" not in llm.attributes  # content is opt-in
    assert spans["node inner"].status.status_code.name == "ERROR"
    assert (llm.end_time - llm.start_time) / 1e9 == pytest.approx(0.05, abs=1e-6)


def test_otel_exporter_can_capture_content() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace = make_trace("ok", llm=[make_llm()], tools=[ToolCall(name="t", args={"a": 1}, result=2)])
    OTelExporter(provider, capture_content=True)(trace)
    attrs = {s.name: s.attributes for s in memory.get_finished_spans()}
    assert attrs["llm m"]["gen_ai.completion"] == "hello"
    assert attrs["tool t"]["tool.args"] == "{'a': 1}"


def test_otel_exporter_requires_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    with pytest.raises(ImportError, match="evalkit-agents\\[otel\\]"):
        OTelExporter()


def test_package_exports_resolve_lazily() -> None:
    import evalkit

    for name in evalkit.__all__:
        assert getattr(evalkit, name) is not None
    assert "run_eval" in dir(evalkit)
    with pytest.raises(AttributeError, match="no attribute"):
        evalkit.does_not_exist  # noqa: B018
