"""Groundedness: is every claim in the answer supported by the retrieved context?"""

from __future__ import annotations

from typing import ClassVar

from evalkit.dataset import Case
from evalkit.judge import Judge, JudgeError, ask_json
from evalkit.scorers.base import Score, Scorer
from evalkit.scorers.text import content_tokens, split_sentences
from evalkit.trace import Trace

JUDGE_PROMPT = """You are checking whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Break the answer into individual factual claims. For each claim decide whether the context
supports it. Do not use outside knowledge.
Reply with JSON only:
{{"claims": [{{"claim": "<text>", "supported": true|false, "evidence": "<quote or empty>"}}]}}"""

ABSTENTIONS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "cannot answer",
    "can't answer",
    "no information",
)


class GroundednessScorer(Scorer):
    """Score = supported claims / total claims; unsupported claims are listed in the details.

    Context comes from ``case.context`` when given, otherwise from what the agent actually
    retrieved (retriever steps and declared context tools).

    Without a judge a lexical heuristic is used: a claim is supported when at least
    ``support_threshold`` of its content words appear in the context or in the user's own
    question (words the user supplied need no evidence). It is
    deliberately lenient (it cannot tell that two true facts were combined wrongly), so use
    a judge when you need strict entailment.
    Answers that abstain ("I don't know") make no claims and score 1.0.
    """

    default_name: ClassVar[str] = "groundedness"
    default_threshold: ClassVar[float] = 0.8

    def __init__(
        self,
        *,
        judge: Judge | None = None,
        support_threshold: float = 0.6,
        use_case_context: bool = True,
        name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.judge = judge
        self.support_threshold = support_threshold
        self.use_case_context = use_case_context

    def _context(self, case: Case, trace: Trace) -> list[str]:
        if self.use_case_context and case.context:
            return list(case.context)
        return trace.retrieved_context()

    def score(self, case: Case, trace: Trace) -> Score:
        answer = trace.answer.strip()
        if not answer:
            return self.skip("agent produced no answer")
        context = self._context(case, trace)
        if not context:
            return self.skip("no context available (none in the case, none retrieved)")
        if any(phrase in answer.lower() for phrase in ABSTENTIONS):
            return self.result(1.0, "answer abstains; it makes no unsupported claims", claims=[])
        if self.judge is not None:
            return self._judge_score(answer, context)
        return self._heuristic_score(answer, context, case.input_text)

    def _heuristic_score(self, answer: str, context: list[str], question: str) -> Score:
        claims = split_sentences(answer) or [answer]
        vocabulary = {t for c in [*context, question] for t in content_tokens(c)}
        rows: list[dict[str, object]] = []
        for claim in claims:
            words = set(content_tokens(claim))
            best = len(words & vocabulary) / len(words) if words else 1.0
            rows.append(
                {
                    "claim": claim,
                    "supported": best >= self.support_threshold,
                    "coverage": round(best, 3),
                }
            )
        return self._summarize(rows, mode="heuristic")

    def _judge_score(self, answer: str, context: list[str]) -> Score:
        assert self.judge is not None
        prompt = JUDGE_PROMPT.format(context="\n---\n".join(context), answer=answer)
        try:
            verdict = ask_json(self.judge, prompt)
            raw = verdict["claims"]
            rows = [
                {
                    "claim": str(c["claim"]),
                    "supported": bool(c["supported"]),
                    "evidence": str(c.get("evidence", "")),
                }
                for c in raw
            ]
        except (JudgeError, KeyError, TypeError) as exc:
            result = self.result(0.0, f"judge failed: {exc}", mode="judge")
            result.error = str(exc)
            result.passed = False
            return result
        if not rows:
            return self.result(1.0, "judge found no factual claims", mode="judge", claims=[])
        return self._summarize(rows, mode="judge")

    def _summarize(self, rows: list[dict[str, object]], *, mode: str) -> Score:
        unsupported = [str(r["claim"]) for r in rows if not r["supported"]]
        supported = len(rows) - len(unsupported)
        value = supported / len(rows)
        if unsupported:
            reason = f"{supported}/{len(rows)} claims supported; unsupported: " + "; ".join(
                f'"{c}"' for c in unsupported
            )
        else:
            reason = f"all {len(rows)} claims supported by context"
        return self.result(value, reason, mode=mode, claims=rows, unsupported_claims=unsupported)
