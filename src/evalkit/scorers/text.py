"""Small text utilities shared by scorers (no third-party dependencies)."""

from __future__ import annotations

import re
import string
from collections import Counter
from difflib import SequenceMatcher

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation.replace(".", "").replace("%", ""))
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_TRAILING_ZEROS = re.compile(r"(?<=\d)\.0+\b")
STOPWORDS = frozenset(
    (
        "a an the is are was were be been being of in on at to for from by with and or but not "
        "as it its this that these those i you he she we they what which who how do does did can "
        "will would should could has have had there their than then so if about into over under"
    ).split()
)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    Numbers are kept intact: ``$2,940`` and ``2940.0`` both normalise to ``2940``.
    """
    text = _THOUSANDS.sub("", text.lower())
    text = _TRAILING_ZEROS.sub("", _STRAY_DOT.sub("", text.translate(_PUNCT)))
    return " ".join(w for w in text.split() if w not in _ARTICLES)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return float(ta == tb)
    overlap = sum((Counter(ta) & Counter(tb)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(tb), overlap / len(ta)
    return 2 * precision * recall / (precision + recall)


def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like claims, dropping trivial fragments."""
    parts = [p.strip(" -*•\t") for p in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if len(p.split()) >= 3]
