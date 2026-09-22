from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalkit.adapters import FunctionAdapter
from evalkit.assertions import EvalAssertionError, assert_eval
from evalkit.baseline import (
    compare,
    format_regressions,
    load_baseline,
    make_baseline,
    save_baseline,
)
from evalkit.capture import current_recorder
from evalkit.dataset import Case
from evalkit.pricing import ModelPrice, PricingConfig
from evalkit.report import (
    build_report,
    load_report,
    render_html,
    render_terminal,
    write_html,
    write_json,
)
from evalkit.runner import run_eval
from evalkit.scorers import AccuracyScorer, GroundednessScorer, LatencyCostScorer, SafetyScorer
from evalkit.scorers.safety import DEFAULT_RULES
from evalkit.trace import ToolCall

PRICING = PricingConfig(default=ModelPrice(input_per_million=1000, output_per_million=1000))


def agent(question: Any) -> str:
    rec = current_recorder()
    assert rec is not None
    rec.record_tool("search", {"q": question}, result="found")
    rec.record_llm(
        model="m",
        prompt=f"<b>{question}</b>",
        output="answer <script>",
        input_tokens=10,
        output_tokens=5,
        time_to_first_token_s=0.001,
    )
    if question == "crash":
        raise RuntimeError("agent crashed")
    return "Pro costs 49 dollars" if question != "wrong" else "I like turtles"


def make_results(extra: list[Case] | None = None) -> Any:
    cases = [
        Case(
            id="good",
            input="price",
            reference="Pro costs 49 dollars",
            context=["Pro costs 49 dollars"],
            tags=["a"],
            expected_tools=["search"],
        ),
        Case(
            id="bad",
            input="wrong",
            reference="Pro costs 49 dollars",
            context=["Pro costs 49 dollars"],
        ),
        Case(id="boom", input="crash", reference="x"),
        *(extra or []),
    ]
    scorers = [
        AccuracyScorer(),
        GroundednessScorer(),
        SafetyScorer(rules=DEFAULT_RULES),
        LatencyCostScorer(pricing=PRICING),
    ]
    return run_eval(cases, FunctionAdapter(agent, name="demo-agent"), scorers, concurrency=1)


@pytest.fixture(scope="module")
def results() -> Any:
    return make_results()


@pytest.fixture(scope="module")
def report(results: Any) -> dict[str, Any]:
    return build_report(results)


def test_build_report_shape(report: dict[str, Any]) -> None:
    assert report["agent"] == "demo-agent"
    assert report["summary"] == {
        "cases": 3,
        "passed": 1,
        "failed": 2,
        "errors": 1,
        "pass_rate": pytest.approx(1 / 3),
    }
    assert set(report["scorers"]) == {"accuracy", "groundedness", "safety", "latency_cost"}
    assert report["scorers"]["latency_cost"]["metrics"]["latency_p95_s"] >= 0
    assert report["cases"][0]["trace"]["steps"]
    slim = build_report(make_results(), include_traces=False)
    assert "trace" not in slim["cases"][0]
    json.dumps(report)  # fully serialisable


def test_write_and_load_report(tmp_path: Path, report: dict[str, Any]) -> None:
    path = write_json(report, tmp_path / "deep" / "r.json")
    assert load_report(path)["run_id"] == report["run_id"]
    (tmp_path / "bad.json").write_text("[]")
    with pytest.raises(ValueError, match="not an EvalKit report"):
        load_report(tmp_path / "bad.json")


def test_terminal_report(report: dict[str, Any]) -> None:
    text = render_terminal(report)
    for fragment in (
        "EvalKit - demo-agent",
        "accuracy",
        "latency p95",
        "failures (2)",
        "bad",
        "agent error: RuntimeError: agent crashed",
    ):
        assert fragment in text
    assert "\x1b[" in render_terminal(report, color=True)
    assert "and 1 more" in render_terminal(report, max_failures=1)
    clean = dict(report, cases=[c for c in report["cases"] if c["passed"]])
    assert "failures" not in render_terminal(clean)


def test_html_report_is_self_contained_and_escaped(tmp_path: Path, report: dict[str, Any]) -> None:
    html = render_html(report)
    assert html.startswith("<!doctype html>") and "<details class='case'" in html
    assert "demo-agent" in html and "latency p95" in html
    assert "<script>" not in html.split("</style>")[1].split("<main>")[0]
    assert "answer &lt;script&gt;" in html and "&lt;b&gt;price&lt;/b&gt;" in html
    assert "answer <script>" not in html  # model output must be escaped
    assert "http://" not in html and "https://" not in html  # no external assets
    assert "Unsupported claims" in html and "Agent error" in html
    path = write_html(report, tmp_path / "r.html")
    assert path.read_text(encoding="utf-8") == html


def test_html_handles_minimal_report() -> None:
    minimal = {
        "agent": "a",
        "run_id": "r" * 32,
        "created_at": "now",
        "evalkit_version": "0",
        "duration_s": 0.0,
        "summary": {"cases": 0, "passed": 0, "failed": 0, "errors": 0, "pass_rate": 0.0},
        "scorers": {"s": {"mean": None, "pass_rate": None, "n": 0, "skipped": 0, "metrics": {}}},
        "cases": [
            {
                "id": "x",
                "input": {"k": 1},
                "reference": None,
                "tags": [],
                "passed": True,
                "error": None,
                "trace_id": "t",
                "latency_s": 0.1,
                "answer": "",
                "scores": [
                    {
                        "scorer": "s",
                        "score": 1.0,
                        "reason": "r",
                        "passed": True,
                        "skipped": True,
                        "details": {"extra": 1},
                    }
                ],
                "trace": {
                    "steps": [
                        {
                            "id": "1",
                            "name": "n",
                            "kind": "node",
                            "latency_s": 0.0,
                            "inputs": {"a": 1},
                            "outputs": "o",
                            "documents": ["d"],
                            "error": "e",
                            "llm": None,
                            "tool": None,
                        }
                    ]
                },
            }
        ],
    }
    html = render_html(minimal)
    assert "n/a" in html and "(no answer)" in html and "skipped" in html
    minimal["cases"][0]["trace"] = {"steps": []}
    assert "no steps recorded" in render_html(minimal)


def test_html_renders_llm_and_tool_steps(report: dict[str, Any]) -> None:
    html = render_html(report)
    assert "Requested tool calls" not in html
    assert "ttft 0.001s" in html and "Arguments" in html and "Result" in html


# ---- baseline --------------------------------------------------------------------------


def test_make_save_load_baseline(tmp_path: Path, report: dict[str, Any]) -> None:
    base = make_baseline(report)
    assert base["version"] == 1 and base["cases"]["good"]["passed"] is True
    assert "latency_p95_s" in base["metrics"]["latency_cost"]
    path = save_baseline(report, tmp_path / "b" / "baseline.json")
    assert load_baseline(path) == base
    (tmp_path / "bad.json").write_text('{"version": 99}')
    with pytest.raises(ValueError, match="not a version-1"):
        load_baseline(tmp_path / "bad.json")


def worse(report: dict[str, Any], *, drop: float = 0.0) -> dict[str, Any]:
    copy = json.loads(json.dumps(report))
    copy["scorers"]["accuracy"]["mean"] -= drop
    return copy


def test_compare_detects_scorer_regressions(report: dict[str, Any]) -> None:
    base = make_baseline(report)
    assert compare(report, base) == []
    assert compare(worse(report, drop=0.01), base) == []  # inside tolerance
    found = compare(worse(report, drop=0.2), base)
    assert [r.kind for r in found] == ["scorer"] and "accuracy mean fell" in found[0].message
    assert compare(worse(report, drop=0.2), base, max_drop=0.5) == []
    assert "No regressions" in format_regressions([])
    assert "accuracy" in format_regressions(found)


def test_compare_missing_scorer_and_null_means(report: dict[str, Any]) -> None:
    base = make_baseline(report)
    missing = json.loads(json.dumps(report))
    del missing["scorers"]["safety"]
    assert [r.kind for r in compare(missing, base)] == ["missing"]
    nulls = json.loads(json.dumps(report))
    nulls["scorers"]["accuracy"]["mean"] = None
    assert compare(nulls, base) == []


def test_compare_metric_limits(report: dict[str, Any]) -> None:
    base = make_baseline(report)
    slower = json.loads(json.dumps(report))
    slower["scorers"]["latency_cost"]["metrics"]["latency_p95_s"] *= 3
    slower["scorers"]["latency_cost"]["metrics"]["cost_per_request_usd"] *= 3
    assert compare(slower, base) == []  # off by default: latency is noisy in CI
    found = compare(slower, base, max_latency_increase=0.5, max_cost_increase=0.5)
    assert {r.name for r in found} == {
        "latency_cost.latency_p95_s",
        "latency_cost.cost_per_request_usd",
    }
    assert compare(slower, base, max_latency_increase=5.0, max_cost_increase=5.0) == []
    stripped = json.loads(json.dumps(report))
    stripped["scorers"]["latency_cost"]["metrics"] = {}
    assert compare(stripped, base, max_latency_increase=0.1) == []


def test_compare_strict_cases(report: dict[str, Any]) -> None:
    base = make_baseline(report)
    flipped = json.loads(json.dumps(report))
    flipped["cases"][0]["passed"] = False
    assert compare(flipped, base) == []
    found = compare(flipped, base, strict_cases=True)
    assert [r.kind for r in found] == ["case"] and "good" in found[0].message


# ---- assert_eval -----------------------------------------------------------------------


def test_assert_eval_passes_and_fails(results: Any) -> None:
    assert_eval(results, min_accuracy=0.3, min_pass_rate=0.3)
    with pytest.raises(EvalAssertionError, match="accuracy mean") as exc:
        assert_eval(results, min_accuracy=0.99)
    assert isinstance(exc.value, AssertionError)
    with pytest.raises(EvalAssertionError, match="pass rate"):
        assert_eval(results, min_pass_rate=0.99)


def test_assert_eval_accepts_report_dicts(report: dict[str, Any]) -> None:
    assert_eval(report, min_safety=0.9)


def test_assert_eval_metric_limits(results: Any) -> None:
    assert_eval(results, max_latency_p95_s=60, max_cost_per_request_usd=1.0)
    with pytest.raises(EvalAssertionError, match="latency_p95_s"):
        assert_eval(results, max_latency_p95_s=0.0)
    with pytest.raises(EvalAssertionError, match="cost_per_request_usd"):
        assert_eval(results, max_cost_per_request_usd=1e-12)
    bare = run_eval([Case(id="x", input="q")], FunctionAdapter(lambda q: "a"), [AccuracyScorer()])
    with pytest.raises(EvalAssertionError, match="not measured"):
        assert_eval(bare, max_latency_p95_s=1)


def test_assert_eval_argument_errors(results: Any) -> None:
    with pytest.raises(ValueError, match="no scorer named 'nope'"):
        assert_eval(results, min_nope=0.5)
    with pytest.raises(TypeError, match="unexpected argument"):
        assert_eval(results, accuracy=0.5)
    unscored = run_eval(
        [Case(id="x", input="q")], FunctionAdapter(lambda q: "a"), [AccuracyScorer()]
    )
    with pytest.raises(EvalAssertionError, match="no cases were scored"):
        assert_eval(unscored, min_accuracy=0.5)


def test_assert_eval_baseline(results: Any, report: dict[str, Any], tmp_path: Path) -> None:
    path = save_baseline(report, tmp_path / "b.json")
    assert_eval(results, baseline=path)
    assert_eval(results, baseline=make_baseline(report))
    better = make_baseline(report)
    better["scorers"]["accuracy"]["mean"] = 1.0
    with pytest.raises(EvalAssertionError, match="accuracy mean fell"):
        assert_eval(results, baseline=better)


def test_tool_call_model_in_reports() -> None:
    assert ToolCall(name="x").model_dump()["args"] == {}
