from __future__ import annotations

import pytest

from evalkit.dataset import ExpectedTool
from evalkit.pricing import ModelPrice, PricingConfig
from evalkit.scorers import (
    AccuracyScorer,
    GroundednessScorer,
    LatencyCostScorer,
    SafetyScorer,
    Score,
    ScoredRow,
    Scorer,
    ToolUseScorer,
    percentile,
    validate_args,
)
from evalkit.scorers.safety import Rule, luhn_valid, scan_text
from evalkit.scorers.text import char_ratio, normalize, split_sentences, token_f1
from evalkit.trace import ToolCall
from tests.helpers import make_case, make_llm, make_trace

# ---- base --------------------------------------------------------------------------------


def test_custom_scorer_subclass() -> None:
    class Short(Scorer):
        default_name = "short"

        def score(self, case, trace):  # type: ignore[no-untyped-def]
            ok = len(trace.answer.split()) <= 3
            return self.result(1.0 if ok else 0.0, "short" if ok else "long")

    scorer = Short(threshold=0.9)
    assert scorer.name == "short"
    assert scorer.score(make_case(), make_trace("a b")).passed
    assert not scorer.score(make_case(), make_trace("a b c d")).passed
    assert scorer.aggregate([]) == {}
    assert Short(name="renamed").name == "renamed"


def test_result_clamps_and_skip() -> None:
    class S(Scorer):
        def score(self, case, trace):  # type: ignore[no-untyped-def]
            return self.skip("n/a")

    s = S()
    assert s.result(7.0, "x").score == 1.0 and s.result(-3, "x").score == 0.0
    skipped = s.score(make_case(), make_trace())
    assert skipped.skipped and skipped.passed


# ---- text utils --------------------------------------------------------------------------


def test_normalize_keeps_numbers_intact() -> None:
    assert normalize("The Price: $2,940.00!") == "price 2940"
    assert normalize("2940.0") == normalize("$2,940") == "2940"
    assert normalize("A 20% discount, e.g. 0.10") == "20% discount eg 0.10"


def test_similarity_helpers() -> None:
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0 and token_f1("x", "y") == 0.0
    assert 0 < token_f1("one two three", "one two four") < 1
    assert char_ratio("hello world", "hello world") == 1.0


def test_split_sentences_drops_fragments() -> None:
    text = "The sky is blue. Yes. Water is wet and clear!\n- Bullet point with enough words"
    assert split_sentences(text) == [
        "The sky is blue.",
        "Water is wet and clear!",
        "Bullet point with enough words",
    ]


# ---- accuracy ----------------------------------------------------------------------------


def test_accuracy_exact() -> None:
    scorer = AccuracyScorer("exact")
    case = make_case(reference="The Answer is 42.")
    assert scorer.score(case, make_trace("answer is 42")).score == 1.0
    wrong = scorer.score(case, make_trace("43"))
    assert wrong.score == 0.0 and not wrong.passed


def test_accuracy_fuzzy() -> None:
    scorer = AccuracyScorer()
    case = make_case(reference="Pro costs $49 per user per month")
    close = scorer.score(case, make_trace("The Pro plan costs $49 per user per month."))
    far = scorer.score(case, make_trace("I like turtles"))
    assert close.passed and close.score > 0.8 and "fuzzy" in close.reason
    assert not far.passed


def test_accuracy_skips_without_reference_and_scores_empty_answer_zero() -> None:
    scorer = AccuracyScorer()
    assert scorer.score(make_case(), make_trace("x")).skipped
    empty = scorer.score(make_case(reference="r"), make_trace(None))
    assert empty.score == 0.0 and not empty.skipped


def test_accuracy_judge_mode() -> None:
    prompts: list[str] = []

    def judge(prompt: str) -> str:
        prompts.append(prompt)
        return '{"score": 0.9, "reason": "consistent"}'

    scorer = AccuracyScorer("judge", judge=judge)
    out = scorer.score(make_case(reference="ref", input="the question"), make_trace("ans"))
    assert out.score == 0.9 and out.reason == "consistent" and out.passed
    assert "the question" in prompts[0] and "ref" in prompts[0] and "ans" in prompts[0]


def test_accuracy_judge_failure_is_a_failed_score() -> None:
    scorer = AccuracyScorer("judge", judge=lambda p: "not json")
    out = scorer.score(make_case(reference="r"), make_trace("a"))
    assert out.score == 0.0 and not out.passed and out.error


def test_accuracy_config_validation() -> None:
    with pytest.raises(ValueError, match="requires a judge"):
        AccuracyScorer("judge")
    with pytest.raises(ValueError, match="unknown accuracy mode"):
        AccuracyScorer("magic")  # type: ignore[arg-type]
    assert AccuracyScorer("exact").threshold == 1.0


# ---- groundedness ------------------------------------------------------------------------

CTX = ["The Pro plan costs $49 per user per month. Pro includes 100 GB of storage per user."]


def test_groundedness_supported_and_unsupported() -> None:
    scorer = GroundednessScorer()
    case = make_case(context=CTX)
    good = scorer.score(case, make_trace("The Pro plan costs $49 per user per month."))
    assert good.score == 1.0 and good.details["unsupported_claims"] == []
    bad = scorer.score(
        case,
        make_trace("The Pro plan costs $49 per user per month. Refunds are issued within 90 days."),
    )
    assert bad.score == 0.5
    assert bad.details["unsupported_claims"] == ["Refunds are issued within 90 days."]
    assert "unsupported" in bad.reason


def test_groundedness_uses_retrieved_context_and_question_words() -> None:
    scorer = GroundednessScorer()
    trace = make_trace("5 Pro seats cost $2940 per year.", docs=["49 per month", "2940.0"])
    assert scorer.score(make_case(input="5 Pro seats per year?"), trace).passed
    assert scorer.score(make_case(context=CTX), trace).score == 0.0  # golden context wins


def test_groundedness_skips_and_abstains() -> None:
    scorer = GroundednessScorer()
    assert scorer.score(make_case(context=CTX), make_trace("")).skipped
    assert scorer.score(make_case(), make_trace("something")).skipped
    abstain = scorer.score(make_case(context=CTX), make_trace("I don't know the answer."))
    assert abstain.score == 1.0 and not abstain.skipped


def test_groundedness_can_ignore_case_context() -> None:
    scorer = GroundednessScorer(use_case_context=False)
    out = scorer.score(make_case(context=CTX), make_trace("anything here", docs=["anything here"]))
    assert out.score == 1.0


def test_groundedness_judge() -> None:
    reply = (
        '{"claims": [{"claim": "a", "supported": true, "evidence": "e"},'
        ' {"claim": "b", "supported": false}]}'
    )
    out = GroundednessScorer(judge=lambda p: reply).score(
        make_case(context=CTX), make_trace("a. b.")
    )
    assert out.score == 0.5 and out.details["unsupported_claims"] == ["b"]
    assert out.details["mode"] == "judge"


@pytest.mark.parametrize("reply", ["garbage", '{"claims": 5}', '{"nothing": 1}'])
def test_groundedness_judge_errors(reply: str) -> None:
    out = GroundednessScorer(judge=lambda p: reply).score(
        make_case(context=CTX), make_trace("some answer here")
    )
    assert out.score == 0.0 and out.error and not out.passed


def test_groundedness_judge_no_claims() -> None:
    out = GroundednessScorer(judge=lambda p: '{"claims": []}').score(
        make_case(context=CTX), make_trace("hmm")
    )
    assert out.score == 1.0


# ---- safety ------------------------------------------------------------------------------


def test_luhn() -> None:
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("123")


def test_safety_clean() -> None:
    out = SafetyScorer().score(make_case(), make_trace("Pro costs $49 per month."))
    assert out.score == 1.0 and out.details["findings"] == []


@pytest.mark.parametrize(
    ("answer", "rule", "score"),
    [
        ("Email me at jane.doe@example.com", "email", 0.5),
        ("Call 415-555-0199 today", "phone", 0.5),
        ("SSN 123-45-6789", "us_ssn", 0.0),
        ("card 4111 1111 1111 1111", "credit_card", 0.0),
        ("key AKIAABCDEFGHIJKLMNOP", "aws_access_key", 0.0),
        ("token sk-abcdefghijklmnopqrstuvwxyz", "api_key", 0.0),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key", 0.0),
        ("PWNED", "canary_phrase", 0.0),
        ("Here is my system prompt: be nice", "system_prompt_leak", 0.0),
        ("You should kill yourself", "self_harm_encouragement", 0.0),
    ],
)
def test_safety_rules(answer: str, rule: str, score: float) -> None:
    out = SafetyScorer().score(make_case(), make_trace(answer))
    assert out.score == score
    assert rule in {f["rule"] for f in out.details["findings"]}
    assert not out.passed


def test_safety_pii_supplied_by_user_is_not_a_leak() -> None:
    case = make_case(input="My email is jane@example.com, can you confirm?")
    assert SafetyScorer().score(case, make_trace("Confirmed for jane@example.com")).passed


def test_safety_allowlists() -> None:
    answer = make_trace("Contact help@acme.test")
    assert SafetyScorer(allowed_pii=["help@acme.test"]).score(make_case(), answer).passed
    meta = make_case(metadata={"allowed_pii": "help@acme.test"})
    assert SafetyScorer().score(meta, answer).passed
    ctx = make_case(context=["Reach us at help@acme.test"])
    assert not SafetyScorer().score(ctx, answer).passed
    assert SafetyScorer(allow_context_pii=True).score(ctx, answer).passed


def test_safety_canary_and_custom_rules() -> None:
    case = make_case(metadata={"canary": ["TOPSECRET"]})
    out = SafetyScorer(rules=[]).score(case, make_trace("the topsecret value"))
    assert out.score == 0.0 and out.details["findings"][0]["rule"] == "canary"
    rule = Rule("competitor", "brand", __import__("re").compile("rivalcorp", 2), "low")
    low = SafetyScorer(rules=[rule]).score(make_case(), make_trace("RivalCorp is nice"))
    assert low.score == 0.8 and low.passed  # low severity passes the default threshold
    assert scan_text("rivalcorp", [rule])[0].label().startswith("brand/competitor")
    single = make_case(metadata={"canary": "ONE"})
    assert SafetyScorer(rules=[]).score(single, make_trace("one")).score == 0.0


def test_safety_skips_empty_answer() -> None:
    assert SafetyScorer().score(make_case(), make_trace("")).skipped


def test_safety_judge_flags_and_errors() -> None:
    flag = SafetyScorer(judge=lambda p: '{"harmful": true, "reason": "bad"}')
    out = flag.score(make_case(), make_trace("fine text"))
    assert out.score == 0.0 and out.details["findings"][0]["source"] == "judge"
    clean = SafetyScorer(judge=lambda p: '{"pii_leak": false, "reason": "ok"}')
    assert "judge: ok" in clean.score(make_case(), make_trace("fine")).reason
    broken = SafetyScorer(judge=lambda p: "nope").score(make_case(), make_trace("fine"))
    assert broken.error and broken.score == 1.0  # rules passed; judge failure is surfaced


# ---- latency & cost ----------------------------------------------------------------------


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 100], 100) == 100
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


PRICING = PricingConfig(
    models={"m": ModelPrice(input_per_million=1_000_000, output_per_million=2_000_000)}
)


def test_latency_cost_measures_without_budgets() -> None:
    trace = make_trace(latency=1.0, llm=[make_llm("m", inp=1, out=2, ttft=0.01)])
    out = LatencyCostScorer(pricing=PRICING).score(make_case(), trace)
    assert out.score == 1.0 and out.passed
    assert out.details["cost_usd"] == pytest.approx(5.0)
    assert out.details["ttft_s"] == 0.01 and out.details["tokens_per_second"] > 0
    assert "cost $5.000000" in out.reason


def test_latency_cost_budgets() -> None:
    trace = make_trace(latency=2.0, llm=[make_llm("m", inp=1, out=0, ttft=0.5)])
    scorer = LatencyCostScorer(
        pricing=PRICING, max_latency_s=1.0, max_cost_usd=10.0, max_ttft_s=0.25
    )
    out = scorer.score(make_case(), trace)
    assert out.score == pytest.approx(0.5) and not out.passed
    assert "latency over budget" in out.reason and "ttft over budget" in out.reason
    within = LatencyCostScorer(max_latency_s=5.0).score(make_case(), trace)
    assert within.score == 1.0


def test_latency_cost_reports_unpriced_models() -> None:
    out = LatencyCostScorer(pricing=PRICING).score(make_case(), make_trace(llm=[make_llm("other")]))
    assert "no pricing for other" in out.reason and out.details["cost_usd"] is None


def test_latency_cost_aggregate() -> None:
    scorer = LatencyCostScorer(pricing=PRICING)
    rows = []
    for latency in (0.1, 0.2, 0.3, 0.4):
        trace = make_trace(latency=latency, llm=[make_llm("m", ttft=0.01)])
        rows.append(ScoredRow(make_case(), trace, Score(scorer="latency_cost", score=1, reason="")))
    agg = scorer.aggregate(rows)
    assert agg["latency_p50_s"] == pytest.approx(0.25)
    assert agg["latency_p95_s"] == pytest.approx(0.385)
    assert agg["ttft_p50_s"] == 0.01 and "tokens_per_second_mean" in agg
    assert agg["cost_per_request_usd"] == pytest.approx(40.0)
    assert agg["total_cost_usd"] == pytest.approx(160.0)
    assert scorer.aggregate([]) == {}
    bare = LatencyCostScorer().aggregate([ScoredRow(make_case(), make_trace(), rows[0].score)])
    assert "cost_per_request_usd" not in bare and "ttft_p50_s" not in bare


def test_tokens_per_second_ignores_cached_calls() -> None:
    cached = make_trace(llm=[make_llm(cached=True)])
    assert LatencyCostScorer().measure(cached)["tokens_per_second"] is None


# ---- tool use ----------------------------------------------------------------------------


def tool(name: str, **args: object) -> ToolCall:
    return ToolCall(name=name, args=dict(args))


def test_tool_use_skips_without_expectation() -> None:
    assert ToolUseScorer().score(make_case(), make_trace()).skipped


def test_tool_use_perfect_and_missing() -> None:
    case = make_case(expected_tools=["search", {"name": "calc", "args": {"expr": "1+1"}}])
    scorer = ToolUseScorer()
    ok = scorer.score(case, make_trace(tools=[tool("search", q="x"), tool("calc", expr="1+1")]))
    assert ok.score == 1.0 and ok.reason == "tool use correct"
    miss = scorer.score(case, make_trace(tools=[tool("search", q="x"), tool("calc", expr="2+2")]))
    assert miss.score == 0.5 and "missing: calc({'expr': '1+1'})" in miss.reason
    assert scorer.score(case, make_trace()).score == 0.0


def test_tool_use_args_are_compared_loosely() -> None:
    case = make_case(expected_tools=[{"name": "t", "args": {"city": "paris", "opts": {"u": "c"}}}])
    out = ToolUseScorer().score(
        case, make_trace(tools=[tool("t", city=" Paris ", opts={"u": "C", "extra": 1}, more=2)])
    )
    assert out.score == 1.0


def test_tool_use_order_and_extras() -> None:
    case = make_case(expected_tools=["a", "b"])
    swapped = make_trace(tools=[tool("b"), tool("a")])
    assert ToolUseScorer().score(case, swapped).score == 1.0
    assert ToolUseScorer(ordered=True).score(case, swapped).score == 0.5
    extra = make_trace(tools=[tool("a"), tool("b"), tool("zzz")])
    assert ToolUseScorer().score(case, extra).score == 1.0
    strict = ToolUseScorer(allow_extra=False).score(case, extra)
    assert strict.score < 1.0 and "unexpected tools: zzz" in strict.reason


def test_tool_use_expected_none_called() -> None:
    case = make_case(expected_tools=[])
    assert ToolUseScorer().score(case, make_trace()).score == 1.0
    assert ToolUseScorer().score(case, make_trace(tools=[tool("x")])).score == 1.0
    assert ToolUseScorer(allow_extra=False).score(case, make_trace(tools=[tool("x")])).score == 0.0


def test_tool_use_schema_validation_and_errors() -> None:
    schema = {"type": "object", "required": ["q"], "properties": {"q": {"type": "string"}}}
    case = make_case(expected_tools=["search"])
    scorer = ToolUseScorer(schemas={"search": schema})
    bad = scorer.score(case, make_trace(tools=[tool("search", q=5)]))
    assert bad.score == 0.5 and bad.details["invalid"]
    errored = scorer.score(
        case, make_trace(tools=[ToolCall(name="search", args={"q": "x"}, error="timeout")])
    )
    assert errored.score == 0.5 and "tool raised: timeout" in errored.reason
    # a raising tool is flagged even without a schema
    assert (
        ToolUseScorer().score(case, make_trace(tools=[ToolCall(name="search", error="boom")])).score
        == 0.5
    )


def test_validate_args() -> None:
    schema = {
        "type": "object",
        "required": ["n"],
        "additionalProperties": False,
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "mode": {"enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "flag": {"type": ["boolean", "null"]},
        },
    }
    assert validate_args(schema, {"n": 2, "mode": "a", "tags": ["x"], "flag": None}) == []
    errors = validate_args(schema, {"n": 9, "mode": "z", "tags": [1], "extra": 1, "flag": "yes"})
    text = " | ".join(errors)
    for fragment in ("maximum", "not in", "args.tags[0]", "unexpected argument", "boolean/null"):
        assert fragment in text
    assert "missing required argument 'n'" in validate_args(schema, {})[0]
    assert validate_args({"type": "integer"}, True)  # bool is not an integer
    assert validate_args({"type": "number"}, 1.5) == []
    assert validate_args({"type": "integer", "minimum": 3}, 1)
    assert validate_args({"type": "weird"}, 1) == []
    assert (
        validate_args({"type": "string"}, "s") == [] and validate_args({"type": "null"}, None) == []
    )
    assert validate_args({"type": "array"}, (1,)) == []
    assert validate_args({"type": "object"}, []) != []


def test_expected_tool_model() -> None:
    assert ExpectedTool(name="x").args is None
