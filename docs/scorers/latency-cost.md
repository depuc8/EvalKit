# Latency & cost

Measures how fast and how expensive each request was, and publishes run-level percentiles.

```python
LatencyCostScorer(pricing=PricingConfig.load("pricing.json"),
                  max_latency_s=5.0, max_cost_usd=0.02, max_ttft_s=1.0)
```

## Per-case measurements (`score.details`)

| Key | Meaning |
| --- | --- |
| `latency_s` | Wall-clock time of the whole run |
| `ttft_s` | Time to first token of the first *streamed* LLM call; `None` if nothing streamed |
| `tokens_per_second` | Output tokens / generation time over LLM calls (excludes replay-cached calls) |
| `input_tokens`, `output_tokens` | Summed over all LLM calls |
| `cost_usd` | From the [pricing config](../running.md#pricing-config); `None` if no call was priced |

## Budgets

Without budgets the score is always 1.0 and the numbers are informational. With a budget the
score is the worst `budget / actual` ratio (capped at 1.0), and the case fails when any budget
is exceeded, with the overage in the reason. CLI: `--max-latency`, `--max-cost`.

## Run-level metrics

Reported under `latency_cost` in the report, usable in `assert_eval` and baselines:

`latency_p50_s`, `latency_p95_s`, `ttft_p50_s`, `ttft_p95_s`, `tokens_per_second_mean`,
`input_tokens_mean`, `output_tokens_mean`, `cost_per_request_usd`, `total_cost_usd`.

```python
assert_eval(results, max_latency_p95_s=5.0, max_cost_per_request_usd=0.02)
```

!!! note
    Latency includes everything in the run, including tool calls and graph overhead, and depends
    on `--concurrency` (more parallel cases means more contention). Compare latency only between
    runs with the same settings and hardware - and in CI prefer cost and token metrics, which are
    stable, over latency.
