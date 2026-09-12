"""Scorer base class and the :class:`Score` result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from evalkit.dataset import Case
from evalkit.trace import Trace


class Score(BaseModel):
    """The outcome of one scorer on one case."""

    scorer: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    passed: bool = True
    skipped: bool = False
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ScoredRow:
    """A case with its trace and score; the input to :meth:`Scorer.aggregate`."""

    case: Case
    trace: Trace
    score: Score


class Scorer(ABC):
    """Base class for scorers.

    Subclass it, set :attr:`default_name`, and implement :meth:`score`::

        class ShortAnswer(Scorer):
            default_name = "short_answer"

            def score(self, case, trace):
                ok = len(trace.answer.split()) <= 50
                return self.result(1.0 if ok else 0.0, "answer is short" if ok else "too long")
    """

    default_name: ClassVar[str] = "scorer"
    default_threshold: ClassVar[float] = 0.5

    def __init__(self, *, name: str | None = None, threshold: float | None = None) -> None:
        self.name = name or self.default_name
        self.threshold = self.default_threshold if threshold is None else threshold

    @abstractmethod
    def score(self, case: Case, trace: Trace) -> Score:
        """Score one run of one case."""

    def aggregate(self, rows: Sequence[ScoredRow]) -> dict[str, float]:
        """Extra run-level metrics (e.g. percentiles) computed over all non-skipped rows."""
        return {}

    # helpers for subclasses ------------------------------------------------
    def result(self, score: float, reason: str, **details: Any) -> Score:
        """Build a :class:`Score`; ``passed`` is ``score >= threshold``."""
        value = max(0.0, min(1.0, score))
        return Score(
            scorer=self.name,
            score=value,
            reason=reason,
            passed=value >= self.threshold,
            details=details,
        )

    def skip(self, reason: str) -> Score:
        """Mark the scorer as not applicable to this case (excluded from averages)."""
        return Score(scorer=self.name, score=1.0, reason=reason, passed=True, skipped=True)
