# Accuracy

Compares the agent's final answer with `case.reference`. Skipped when the case has no reference.

```python
AccuracyScorer("exact")                  # normalised string equality
AccuracyScorer("fuzzy")                  # default
AccuracyScorer("judge", judge=my_judge)  # LLM-as-judge
```

## Modes

**`exact`** - 1.0 if the answers are equal after normalisation, else 0.0. Normalisation
lowercases, strips punctuation and the articles *a/an/the*, collapses whitespace, and keeps
numbers intact (`$2,940`, `2940.0` and `2940` are equal). Threshold 1.0.

**`fuzzy`** - `max(character similarity, token F1)` on normalised text, a number in [0, 1].
Token F1 rewards sharing the right words in any order; character similarity (difflib) rewards
near-identical wording. Default threshold 0.6.

!!! warning "Fuzzy is not semantic"
    "No refunds after 30 days" and "Refunds after 30 days" score high. Use `judge` mode when
    negation or numbers matter, and prefer references that are close to how your agent phrases
    answers.

**`judge`** - the judge sees the question, reference and answer and returns
`{"score": 0..1, "reason": "..."}`. It is told that extra correct detail is fine, while
contradictions and missing key facts are not.

## Details

`details` holds the mode and, for fuzzy, `char_ratio` and `token_f1`. A run that produced no
answer scores 0.0 ("agent produced no answer").
