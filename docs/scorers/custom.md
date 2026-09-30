# Writing a custom scorer

Subclass `Scorer`, give it a name, and implement `score(case, trace)`.

```python
# my_pkg/scorers.py
from evalkit import Scorer


class PoliteTone(Scorer):
    default_name = "polite_tone"       # shown in reports; used by --fail-under / assert_eval
    default_threshold = 0.5            # score >= threshold counts as passed

    def score(self, case, trace):
        answer = trace.answer.lower()
        if not answer:
            return self.skip("no answer to judge")
        rude = [w for w in ("stupid", "obviously", "just read the docs") if w in answer]
        if rude:
            return self.result(0.0, f"rude phrasing: {', '.join(rude)}", phrases=rude)
        return self.result(1.0, "no rude phrasing found")
```

Use it from Python (`run_eval(cases, adapter, [PoliteTone()])`) or the CLI, which instantiates
the class with no arguments:

```bash
evalkit run --dataset golden.jsonl --agent my_pkg.agent:graph \
  --scorers accuracy,my_pkg.scorers:PoliteTone --fail-under polite_tone=0.95
```

## What you can use

- `case` - `.id`, `.input`, `.input_text`, `.reference`, `.expected_tools`, `.context`,
  `.tags`, `.metadata`.
- `trace` - `.answer` (final output as text), `.output`, `.error`, `.latency_s`, `.steps`,
  `.llm_calls`, `.tool_calls`, `.usage`, `.retrieved_context()`.

## Helpers

| Call | Effect |
| --- | --- |
| `self.result(score, reason, **details)` | Builds the `Score`; clamps to [0, 1]; sets `passed` from the threshold; `details` are stored |
| `self.skip(reason)` | Marks the scorer not applicable; excluded from means |
| `self.threshold`, `self.name` | Resolved from constructor args or class defaults |

Return a **reason that explains the score** - it is what you will read at 2 a.m. when a case
fails. Put machine-readable evidence in `details`.

## Constructor options and judges

```python
class Concise(Scorer):
    default_name = "concise"

    def __init__(self, max_words: int = 80, *, name=None, threshold=None):
        super().__init__(name=name, threshold=threshold)
        self.max_words = max_words
```

For LLM-judged scorers accept a `judge` callable and use `evalkit.judge.ask_json(judge, prompt)`,
which parses the reply as JSON and raises `JudgeError` on failure - catch it and return a failing
score with `error` set rather than a misleading 1.0 (see `AccuracyScorer._judge` for the pattern).

## Run-level metrics

```python
def aggregate(self, rows):          # rows: list[ScoredRow(case, trace, score)]
    words = [len(r.trace.answer.split()) for r in rows]
    return {"mean_words": sum(words) / len(words)} if words else {}
```

Returned metrics appear in the report under the scorer's name.

## Errors are contained

If `score` raises, the runner records a failed score (`scorer raised ValueError: ...`) for that
case and carries on, so one bad scorer cannot abort a long run - but fix it, because the case
will count as failing.
