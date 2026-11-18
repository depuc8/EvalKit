# Quickstart

## Install

```bash
pip install "evalkit-agents[langgraph]"
```

## Run the bundled example

The repository contains a sample LangGraph RAG agent and a 50-case golden dataset. The agent
uses a deterministic offline model, so there is nothing to configure.

```bash
git clone https://github.com/depuc8/EvalKit && cd EvalKit
pip install -e ".[dev]"
evalkit run \
  --dataset examples/rag_agent/golden.jsonl \
  --agent examples.rag_agent.agent:agent \
  --pricing examples/rag_agent/pricing.json
```

You get a terminal summary and three files in `evalkit_results/`:

| File | Contents |
| --- | --- |
| `report.html` | Static report; click a case to see scores, reasons and the step-by-step trace |
| `report.json` | The same data, machine-readable (input to baselines and CI) |
| `traces.jsonl` | One full trace per line; what `evalkit replay` reads |

## Evaluate your own LangGraph agent

1. Write a golden set (`golden.jsonl`), one case per line:

    ```json
    {"id": "refund-1", "input": "Can I get a refund?", "reference": "Refunds are available within 30 days.", "expected_tools": ["search_kb"]}
    ```

2. Point EvalKit at a compiled graph in your code (`module:attribute`):

    ```bash
    evalkit run --dataset golden.jsonl --agent my_pkg.agent:graph
    ```

    String inputs are passed as `{"question": ...}`. If your graph's state uses other keys, wrap
    it explicitly:

    ```python
    # my_pkg/eval_agent.py
    from evalkit.adapters.langgraph import LangGraphAdapter
    from my_pkg.agent import graph

    agent = LangGraphAdapter(graph, input_key="messages", context_tools={"search_docs"})
    ```

    ```bash
    evalkit run --dataset golden.jsonl --agent my_pkg.eval_agent:agent
    ```

3. Read `evalkit_results/report.html`, then [replay](replay.md) the failures.

Next: pick [scorers](scorers/index.md), set up [CI](ci.md).
