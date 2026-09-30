# Traces

A **trace** is the structured record of one agent run. `evalkit run` stores one per case in
`traces.jsonl` (one JSON object per line).

```text
Trace
├─ trace_id, agent, case_id, start_time, latency_s
├─ input, output, error
└─ steps[]
   ├─ id, parent_id, name, kind (node | llm | tool | retriever | custom)
   ├─ start_time, latency_s, inputs, outputs, error
   ├─ llm:  model, prompt (messages), output, tool_calls, usage (input/output tokens),
   │        time_to_first_token_s, cached
   ├─ tool: name, args, result, error
   └─ documents: retrieved passages (retriever steps and declared context tools)
```

`parent_id` preserves nesting, e.g. an LLM call inside the `generate` node.

## LangGraph

`LangGraphAdapter` attaches `EvalKitCallback` for you. To record a graph yourself:

```python
from evalkit.capture import TraceRecorder
from evalkit.adapters.langgraph import EvalKitCallback

rec = TraceRecorder(agent="my-graph", input=question)
graph.invoke({"question": question}, {"callbacks": [EvalKitCallback(rec)]})
trace = rec.finish(output=...)
```

- Graph **nodes** are recorded as `node` steps; inner runnables are folded into them.
- **Time to first token** is only available for streaming models. The adapter streams the graph
  in `messages` mode so streaming-capable models report it; pass `measure_ttft=False` to disable.
- **Retrieved context** comes from LangChain retrievers automatically. For retrieval done in a
  tool, list the tool names: `LangGraphAdapter(graph, context_tools={"search_docs"})`.

## Plain Python agents

```python
from evalkit import traced, current_recorder

@traced(store="traces.jsonl")            # works for sync and async functions
def my_agent(question: str) -> str:
    rec = current_recorder()
    rec.record_tool("search", {"q": question}, result=["doc 1", "doc 2"])
    rec.record_llm(model="gpt-x", prompt="...", output="...", input_tokens=120, output_tokens=40,
                   latency_s=0.8)
    return "answer"
```

`rec.step("name", kind)` is a context manager for timing arbitrary blocks. `trace_run(...)`
is the same thing as a `with` block when a decorator is inconvenient.

## OpenTelemetry

```python
from evalkit.otel import OTelExporter          # pip install "evalkit-agents[otel]"

exporter = OTelExporter()                      # uses the global tracer provider
run_eval(cases, adapter, scorers, exporters=[exporter])     # or: evalkit run --otel
```

Spans keep the original start/end times and nesting, and carry `gen_ai.request.model` and
`gen_ai.usage.*` attributes. Prompts, completions and tool arguments are only attached with
`OTelExporter(capture_content=True)` because they may contain sensitive data.
