# EvalKit

EvalKit evaluates and debugs LLM agents.

- **Trace capture** - every run becomes a structured [trace](traces.md): inputs, each graph node,
  tool calls with arguments and results, LLM prompts and outputs, token counts and per-step latency.
- **[Scorers](scorers/index.md)** - accuracy, groundedness, safety, latency & cost, tool use. Write
  your own by [subclassing `Scorer`](scorers/custom.md).
- **[Datasets](datasets.md)** - golden sets in JSONL or CSV.
- **[Runner and reports](running.md)** - concurrent runs; terminal, JSON and static HTML reports.
- **[Replay](replay.md)** - re-run a failed case, swap the prompt or model at one step, diff the runs.
- **[CI](ci.md)** - a pytest plugin and a baseline check that fails the build on regressions.

LangGraph is supported first. The [adapter interface](adapters.md) is small so other frameworks
can be added.

Start with the [quickstart](quickstart.md).
