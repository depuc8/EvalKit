# Framework adapters

An adapter tells EvalKit how to run an agent from some framework and how to capture what
happens. LangGraph ships first; adding ADK, CrewAI or anything else means writing one small class.

```python
from evalkit.adapters import AgentAdapter
from evalkit.capture import TraceRecorder

class MyFrameworkAdapter(AgentAdapter):
    name = "my-framework"

    def __init__(self, agent):
        self.agent = agent

    def invoke(self, input, recorder: TraceRecorder):
        # 1. wire the framework's hooks to the recorder (see below)
        # 2. run the agent
        # 3. return the final answer
        return self.agent.run(input, callbacks=[MyCallback(recorder)])
```

Optional overrides:

- `async ainvoke(input, recorder)` - native async path (default runs `invoke` in a thread).
- `context_tools` - names of tools whose results count as retrieved context.

## Recording steps

`TraceRecorder` is thread-safe and framework-neutral. Call it from your framework's hooks:

```python
step_id = recorder.begin_step("search", "tool", inputs={"q": q}, parent_id=parent)
...
recorder.end_step(step_id, outputs=result, tool=ToolCall(name="search", args={"q": q}, result=result))
```

| Need | Call |
| --- | --- |
| Start / finish any step | `begin_step(name, kind, inputs=, parent_id=)` / `end_step(step_id, outputs=, error=, llm=, tool=, documents=)` |
| A finished LLM call | `record_llm(model=, prompt=, output=, input_tokens=, output_tokens=, latency_s=)` |
| A finished tool call | `record_tool(name, args, result=, error=, latency_s=)` |
| Streaming | `mark_first_token(step_id)` on the first token to get time-to-first-token |
| Retrieval | `end_step(..., documents=[...])`, or declare the tool in `context_tools` |

Look at `evalkit/adapters/langgraph.py` (`EvalKitCallback`) for a complete example; it is about
300 lines including every event type.

## Registering

```python
from evalkit.adapters import register_adapter

register_adapter(lambda obj: isinstance(obj, MyAgent), MyFrameworkAdapter)
```

After registration, `evalkit run --agent my_module:agent_object` and `evalkit.adapt(obj)` wrap
objects of that type automatically. Without registering, export an adapter instance
(`agent = MyFrameworkAdapter(my_agent)`) and point `--agent` at it.

## Replay support

To make [replay](replay.md) work for your framework, intercept LLM calls and consult the active
plan:

```python
from evalkit.replay import current_plan

plan = current_plan()
if plan is not None:
    decision = plan.next_decision()
    if decision.cached:                 # answer from the recording
        return decision.cached.output
    if decision.override:               # run live with .model / .system / .prompt changed
        ...
```

`evalkit.adapters.langgraph.ReplayableLLM` is the reference implementation.

## Plain functions

You do not need an adapter for a function `fn(input) -> output`: `--agent module:fn` wraps it in
a `FunctionAdapter`, and inside it `evalkit.current_recorder()` lets you log steps.
