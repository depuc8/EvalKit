# Scorers

A scorer looks at one case and the trace of the agent's run, and returns a **score from 0 to 1**
with a human-readable **reason**:

```python
Score(scorer="accuracy", score=0.83, reason="fuzzy similarity 0.83 (...)", passed=True,
      skipped=False, error=None, details={...})
```

| Field | Meaning |
| --- | --- |
| `score` | 0.0 (worst) to 1.0 (best) |
| `passed` | `score >= scorer.threshold`; a case passes only if every non-skipped score passed |
| `skipped` | The scorer does not apply (e.g. no reference answer). Skipped scores are excluded from averages |
| `error` | Set when a judge call failed; the score is then a failure, never silently 1.0 |
| `details` | Structured evidence (unsupported claims, safety findings, measured latency, ...) |

| Scorer | Question it answers | Needs | Default threshold |
| --- | --- | --- | --- |
| [`accuracy`](accuracy.md) | Does the answer match the reference? | `reference` | 0.6 (1.0 for `exact`) |
| [`groundedness`](groundedness.md) | Is every claim supported by the context? | `context` or retrieved docs | 0.8 |
| [`safety`](safety.md) | PII leak, prompt-injection compliance, harmful content? | - | 0.8 |
| [`latency_cost`](latency-cost.md) | How fast and expensive was it? | pricing for cost | 1.0 |
| [`tool_use`](tool-use.md) | Right tools, valid arguments? | `expected_tools` | 1.0 |

Thresholds are constructor arguments: `AccuracyScorer(threshold=0.9)`.

## Rules vs. LLM judges

Every text scorer works without an LLM, using deterministic rules. These are free and
reproducible, but lexical. Pass a `judge` for semantic checking:

```python
from evalkit.judge import AnthropicJudge
judge = AnthropicJudge("claude-haiku-5-5")      # any (prompt: str) -> str works
AccuracyScorer("judge", judge=judge)
GroundednessScorer(judge=judge)
SafetyScorer(judge=judge)                       # rules still run; the stricter verdict wins
```

Judges must reply with a JSON object; fences and surrounding chatter are tolerated. If a judge
call fails or returns garbage, the score has `error` set and counts as failed.

!!! tip "Calibrate your judge"
    LLM judges are noisy. Spot-check a few judged cases against your own reading before trusting
    a number, and keep the same judge model between baseline and CI runs.

## Run-level metrics

A scorer can also override `aggregate(rows)` to publish metrics over the whole run - the
latency scorer reports p50/p95 this way. They appear in the report under that scorer.

## Using scorers directly

```python
scorer = GroundednessScorer()
score = scorer.score(case, trace)
```

Or [write your own](custom.md).
