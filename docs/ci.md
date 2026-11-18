# CI integration

Three layers, use whichever fits:

1. `assert_eval` / the pytest plugin - absolute quality bars inside your test suite.
2. `--fail-under` - the same bars from the command line.
3. A stored **baseline** - fail only when results get *worse* than a known-good run.

## pytest

The plugin registers itself when `evalkit-agents` is installed.

```python
from evalkit import AccuracyScorer, GroundednessScorer, adapt, load_dataset, run_eval

def test_agent_quality(assert_eval):          # fixture; or: from evalkit import assert_eval
    results = run_eval(load_dataset("golden.jsonl"), adapt(graph),
                       [AccuracyScorer(), GroundednessScorer()])
    assert_eval(results, min_accuracy=0.85, min_groundedness=0.9)
```

`assert_eval(results, ...)` raises `EvalAssertionError` (an `AssertionError`) listing every
violated threshold:

| Argument | Requires |
| --- | --- |
| `min_<scorer>=x` | That scorer's mean score >= x (`min_accuracy`, `min_safety`, or a custom scorer's name) |
| `min_pass_rate=x` | Fraction of fully passing cases >= x |
| `max_latency_p95_s=x`, `max_cost_per_request_usd=x` | Needs the `latency_cost` scorer (and pricing for cost) |
| `baseline=path_or_dict`, `max_drop=0.02` | No scorer mean fell more than `max_drop` vs. the baseline |

Unknown scorer names raise `ValueError`, so a typo cannot silently pass.

## Baselines

```bash
# once, on a known-good commit - commit the file:
evalkit run --dataset golden.jsonl --agent my_pkg.agent:graph --save-baseline baseline.json

# in CI, on every PR:
evalkit run --dataset golden.jsonl --agent my_pkg.agent:graph --baseline baseline.json \
  --max-regression 0.02
```

The run exits **1** if any scorer's mean falls by more than `--max-regression` (absolute,
default 0.02) or a scorer present in the baseline is missing. Optional, off by default:

| Flag | Meaning |
| --- | --- |
| `--max-cost-increase 0.10` | Fail if cost per request rises more than 10% |
| `--max-latency-increase 0.25` | Fail if p95 latency rises more than 25% (noisy in CI; use generously) |
| `--strict-cases` | Fail on any case that passed in the baseline but fails now |

`evalkit baseline report.json -o baseline.json` makes a baseline from an existing report, and
`evalkit compare baseline.json report.json` checks one offline. Update the baseline deliberately,
in a reviewed commit, when you accept a change in quality.

!!! warning "Keep CI runs reproducible"
    With real LLMs, scores vary between runs. Use `temperature=0`, enough cases that one flipped
    case is within `--max-regression` (with 50 cases, one case is 0.02), and the same judge model
    for baseline and CI.

## GitHub Action

[`examples/github-actions/evalkit-regression.yml`](https://github.com/depuc8/EvalKit/blob/main/examples/github-actions/evalkit-regression.yml)
is a ready-to-copy workflow: it installs your project, runs the evaluation against the stored
baseline, writes the summary to the job page, uploads the HTML report as an artifact, and fails
the build on regressions.

```yaml
- name: Evaluate against baseline
  run: |
    evalkit run \
      --dataset evals/golden.jsonl \
      --agent my_pkg.agent:graph \
      --baseline evals/baseline.json \
      --max-regression 0.02 \
      --fail-under accuracy=0.80 -q
```

This repository runs the same workflow against its bundled example
(`.github/workflows/eval-example.yml`).
