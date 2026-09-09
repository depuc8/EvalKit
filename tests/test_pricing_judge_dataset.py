from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalkit.dataset import Case, DatasetError, filter_cases, load_csv, load_dataset, load_jsonl
from evalkit.judge import (
    AnthropicJudge,
    JudgeError,
    ask_json,
    clamp01,
    load_judge,
    parse_json_object,
)
from evalkit.pricing import ModelPrice, PricingConfig
from evalkit.trace import ToolCall
from tests.helpers import make_llm, make_trace

# ---- pricing -----------------------------------------------------------------------------


def test_price_cost() -> None:
    price = ModelPrice(input_per_million=2.0, output_per_million=10.0)
    assert price.cost(make_llm(inp=1_000_000, out=500_000).usage) == pytest.approx(7.0)


def test_pricing_lookup_order() -> None:
    cfg = PricingConfig(
        models={
            "gpt": ModelPrice(input_per_million=1, output_per_million=1),
            "gpt-mini": ModelPrice(input_per_million=0.1, output_per_million=0.1),
        },
        default=ModelPrice(input_per_million=9, output_per_million=9),
    )
    assert cfg.price_for("gpt") is cfg.models["gpt"]
    assert cfg.price_for("gpt-mini-2025") is cfg.models["gpt-mini"]  # longest prefix wins
    assert cfg.price_for("other") is cfg.default
    assert cfg.price_for(None) is cfg.default
    assert PricingConfig().price_for("x") is None


def test_trace_cost_and_unpriced() -> None:
    cfg = PricingConfig(models={"a": ModelPrice(input_per_million=1e6, output_per_million=0)})
    trace = make_trace(llm=[make_llm("a", inp=2), make_llm("b", inp=5)])
    assert cfg.trace_cost(trace) == pytest.approx(2.0)
    assert cfg.unpriced_models(trace) == ["b"]
    assert cfg.trace_cost(make_trace(llm=[make_llm("b")])) is None
    assert cfg.trace_cost(make_trace(tools=[ToolCall(name="t")])) is None


def test_pricing_load_json_and_toml(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text(
        json.dumps({"models": {"m": {"input_per_million": 1, "output_per_million": 2}}})
    )
    (tmp_path / "p.toml").write_text("[models.m]\ninput_per_million = 1\noutput_per_million = 2\n")
    assert PricingConfig.load(tmp_path / "p.json") == PricingConfig.load(tmp_path / "p.toml")


# ---- judge -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', 'sure!\n```json\n{"a": 1}\n```', 'Here: {"a": 1} hope that helps'],
)
def test_parse_json_object_variants(text: str) -> None:
    assert parse_json_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json", "[1, 2]", "{broken"])
def test_parse_json_object_rejects(text: str) -> None:
    with pytest.raises(JudgeError):
        parse_json_object(text)


def test_ask_json_wraps_failures() -> None:
    assert ask_json(lambda p: '{"ok": true}', "x") == {"ok": True}

    def boom(prompt: str) -> str:
        raise ConnectionError("down")

    with pytest.raises(JudgeError, match="ConnectionError"):
        ask_json(boom, "x")

    def judge_error(prompt: str) -> str:
        raise JudgeError("already")

    with pytest.raises(JudgeError, match="already"):
        ask_json(judge_error, "x")


def test_clamp01() -> None:
    assert clamp01(5) == 1.0 and clamp01(-1) == 0.0 and clamp01("0.5") == 0.5
    assert clamp01("junk", default=0.3) == 0.3 and clamp01(None) == 0.0


def test_anthropic_judge_with_injected_client() -> None:
    calls: list[dict[str, Any]] = []

    class Messages:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"score": 1}'), SimpleNamespace()]
            )

    judge = AnthropicJudge("claude-test", client=SimpleNamespace(messages=Messages()))
    assert judge("prompt") == '{"score": 1}'
    assert calls[0]["model"] == "claude-test" and calls[0]["messages"][0]["content"] == "prompt"


def test_anthropic_judge_requires_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(JudgeError, match="anthropic package"):
        AnthropicJudge()


def test_anthropic_judge_builds_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(Anthropic=lambda: SimpleNamespace(messages=None))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert isinstance(load_judge("anthropic"), AnthropicJudge)
    judge = load_judge("anthropic:claude-x")
    assert isinstance(judge, AnthropicJudge) and judge.model == "claude-x"


def test_load_judge_module_spec() -> None:
    assert load_judge("json:dumps") is not None
    for bad in ("nocolon", "no_such_module_xyz:f", "json:nope", "json:decoder"):
        with pytest.raises(JudgeError):
            load_judge(bad)


# ---- dataset -----------------------------------------------------------------------------


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_jsonl_basic(tmp_path: Path) -> None:
    p = write(
        tmp_path / "d.jsonl",
        "# comment\n\n"
        '{"id": "a", "input": "q1", "reference_answer": "r1", "expected_tools": ["search", '
        '{"name": "calc", "args": {"x": 1}}], "context": "single", "tags": ["t"]}\n'
        '{"input": {"question": "q2"}, "expected": "r2"}\n',
    )
    a, b = load_jsonl(p)
    assert a.reference == "r1" and a.context == ["single"] and a.tags == ["t"]
    assert [t.name for t in a.expected_tools or []] == ["search", "calc"]
    assert (a.expected_tools or [])[1].args == {"x": 1}
    assert b.id == "case-002" and b.reference == "r2"
    assert b.input_text == '{"question": "q2"}'


def test_load_jsonl_errors(tmp_path: Path) -> None:
    cases = {
        '{"input": "x"\n': "invalid JSON",
        "[1]\n": "must be a JSON object",
        '{"id": "a"}\n': "missing required field 'input'",
        '{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n': "duplicate case id",
        '{"input": "x", "tags": "oops"}\n': "tags",
    }
    for i, (text, message) in enumerate(cases.items()):
        with pytest.raises(DatasetError, match=message):
            load_jsonl(write(tmp_path / f"bad{i}.jsonl", text))


def test_load_csv(tmp_path: Path) -> None:
    p = write(
        tmp_path / "d.csv",
        "id,input,reference,expected_tools,context,tags,metadata\n"
        'a,What?,Because,"search, calc",one|two,"x,y","{""canary"": ""PWNED""}"\n'
        'b,Plain,,"[{""name"": ""t"", ""args"": {""k"": 1}}]","[""ctx""]",,\n',
    )
    a, b = load_csv(p)
    assert [t.name for t in a.expected_tools or []] == ["search", "calc"]
    assert a.context == ["one", "two"] and a.tags == ["x", "y"]
    assert a.metadata == {"canary": "PWNED"}
    assert b.reference is None and (b.expected_tools or [])[0].args == {"k": 1}
    assert b.context == ["ctx"]


def test_load_csv_bad_json_cell(tmp_path: Path) -> None:
    p = write(tmp_path / "d.csv", 'input,context\nq,"[broken"\n')
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_csv(p)


def test_load_dataset_dispatch(tmp_path: Path) -> None:
    write(tmp_path / "a.jsonl", '{"input": "q"}\n')
    write(tmp_path / "a.csv", "input\nq\n")
    write(tmp_path / "a.txt", "q")
    assert len(load_dataset(tmp_path / "a.jsonl")) == 1
    assert len(load_dataset(tmp_path / "a.csv")) == 1
    with pytest.raises(DatasetError, match="unsupported"):
        load_dataset(tmp_path / "a.txt")
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_filter_cases() -> None:
    cases = [Case(id=str(i), input="q", tags=["a"] if i % 2 else ["b"]) for i in range(6)]
    assert [c.id for c in filter_cases(cases, tags=["a"])] == ["1", "3", "5"]
    assert len(filter_cases(cases, limit=2)) == 2
    assert len(filter_cases(cases)) == 6
