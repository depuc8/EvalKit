# Running evaluations and reports

```bash
evalkit run --dataset golden.jsonl --agent my_pkg.agent:graph [options]
```

`--agent` is `module:attribute` or `path/to/file.py:attribute`. The attribute may be an
`AgentAdapter`, a compiled LangGraph graph (wrapped automatically), or a plain function taking
the case input. The current directory is importable.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scorers` | all five | Comma-separated: `accuracy,groundedness,safety,latency_cost,tool_use` or `module:Class` |
| `--accuracy-mode` | `fuzzy` | `exact`, `fuzzy` or `judge` |
| `--judge` | none | `anthropic[:model]` or `module:callable` |
| `--pricing` | none | Pricing file (`.json`/`.toml`) enabling cost metrics |
| `--max-latency` / `--max-cost` | none | Per-request budgets; exceeding one fails the case |
| `--concurrency`, `-c` | 4 | Cases in flight at once |
| `--timeout` | none | Per-case timeout in seconds; a timed-out case is a failed case |
| `--tag`, `--limit` | | Run a subset |
| `--output-dir`, `-o` | `evalkit_results` | Where reports and traces go |
| `--otel` | off | Export spans through the global OpenTelemetry provider |
| `--fail-under NAME=VALUE` | | Exit 1 if that scorer's mean (or `pass_rate`) is lower; repeatable |
| `--baseline`, `--save-baseline` | | See [CI](ci.md) |

Exit status: `0` success, `1` a threshold or baseline check failed, `2` usage error (bad
dataset, unknown agent, ...). A case where the agent raises an exception is recorded as failed,
with the error in its trace - the run continues.

## Pricing config

```json
{
  "models": {
    "my-model": {"input_per_million": 3.00, "output_per_million": 15.00}
  },
  "default": {"input_per_million": 1.00, "output_per_million": 1.00}
}
```

Models match exactly, then by longest prefix (`my-model-2025-06` uses `my-model`), then
`default`. EvalKit ships no built-in prices because they change; copy them from your provider.

## Reports

- **Terminal** - per-scorer mean / pass rate / counts, performance metrics, and the first failures
  with the reasons they failed.
- **`report.json`** - summary, per-scorer stats, and every case with its scores and full trace.
- **`report.html`** - self-contained (no network, no external assets). Each case expands to show
  input, answer, reference, every score with its reason (unsupported claims, safety findings),
  and a step timeline with prompts, outputs, tool arguments/results and a latency bar.

Re-render without re-running: `evalkit report evalkit_results/report.json --html out.html`.
List traces: `evalkit traces [--failed]`.

## From Python

```python
from evalkit import run_eval, run_eval_async, build_report, render_terminal
results = run_eval(cases, adapter, scorers, concurrency=8, timeout_s=60)
print(render_terminal(build_report(results)))
results.summary()["accuracy"].mean
results.failures        # CaseResult objects: .case, .trace, .scores
```

`run_eval` is safe to call from a running event loop (notebooks); use `run_eval_async` if you prefer.
