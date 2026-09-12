"""Accuracy: compare the final answer with a reference answer."""

from __future__ import annotations

from typing import ClassVar, Literal

from evalkit.dataset import Case
from evalkit.judge import Judge, JudgeError, ask_json, clamp01
from evalkit.scorers.base import Score, Scorer
from evalkit.scorers.text import char_ratio, normalize, token_f1
from evalkit.trace import Trace

AccuracyMode = Literal["exact", "fuzzy", "judge"]

JUDGE_PROMPT = """You are grading an AI assistant's answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

Assistant answer:
{answer}

Decide whether the assistant answer is factually consistent with the reference and answers the
question. Extra correct detail is fine; contradictions or missing key facts are not.
Reply with JSON only: {{"score": <number from 0 to 1>, "reason": "<one sentence>"}}"""


class AccuracyScorer(Scorer):
    """Scores the final answer against ``case.reference``.

    Modes:

    * ``exact`` - 1.0 if the normalised strings are equal, else 0.0.
    * ``fuzzy`` - ``max(character similarity, token F1)`` on normalised text.
    * ``judge`` - an LLM grades the answer (requires ``judge``).
    """

    default_name: ClassVar[str] = "accuracy"
    default_threshold: ClassVar[float] = 0.6

    def __init__(
        self,
        mode: AccuracyMode = "fuzzy",
        *,
        judge: Judge | None = None,
        name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        if mode == "judge" and judge is None:
            raise ValueError("accuracy mode 'judge' requires a judge")
        if mode not in ("exact", "fuzzy", "judge"):
            raise ValueError(f"unknown accuracy mode {mode!r}")
        if threshold is None and mode == "exact":
            threshold = 1.0
        super().__init__(name=name, threshold=threshold)
        self.mode = mode
        self.judge = judge

    def score(self, case: Case, trace: Trace) -> Score:
        if case.reference is None:
            return self.skip("case has no reference answer")
        answer = trace.answer
        if not answer.strip():
            return self.result(0.0, "agent produced no answer", mode=self.mode)
        if self.mode == "exact":
            ok = normalize(answer) == normalize(case.reference)
            return self.result(
                1.0 if ok else 0.0,
                "answer matches reference exactly" if ok else "answer differs from reference",
                mode="exact",
            )
        if self.mode == "fuzzy":
            ratio, f1 = char_ratio(answer, case.reference), token_f1(case.reference, answer)
            value = max(ratio, f1)
            return self.result(
                value,
                f"fuzzy similarity {value:.2f} (char ratio {ratio:.2f}, token F1 {f1:.2f})",
                mode="fuzzy",
                char_ratio=round(ratio, 4),
                token_f1=round(f1, 4),
            )
        return self._judge(case, answer)

    def _judge(self, case: Case, answer: str) -> Score:
        assert self.judge is not None
        assert case.reference is not None
        prompt = JUDGE_PROMPT.format(
            question=case.input_text, reference=case.reference, answer=answer
        )
        try:
            verdict = ask_json(self.judge, prompt)
        except JudgeError as exc:
            result = self.result(0.0, f"judge failed: {exc}", mode="judge")
            result.error = str(exc)
            result.passed = False
            return result
        value = clamp01(verdict.get("score"))
        return self.result(value, str(verdict.get("reason") or "no reason given"), mode="judge")
