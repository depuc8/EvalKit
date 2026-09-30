# Tool-use correctness

Did the agent call the expected tools, with valid arguments? Skipped when the case has no
`expected_tools`.

```json
{"input": "5 Pro seats per year?",
 "expected_tools": ["search_kb", {"name": "calculator", "args": {"expression": "49*5*12"}}]}
```

```python
ToolUseScorer(schemas={"calculator": {"type": "object", "required": ["expression"],
                                      "properties": {"expression": {"type": "string"}}}},
              ordered=False, allow_extra=True)
```

The score is the mean of the checks that apply:

| Check | When | Score |
| --- | --- | --- |
| **match** | `expected_tools` non-empty | Fraction of expectations found among the calls |
| **valid** | A call has a schema in `schemas`, or raised an error | Fraction of such calls that are valid |
| **no extras** | `allow_extra=False` | Fraction of calls to expected tools |

- An expectation with `args` matches a call whose arguments **contain** those values. Strings
  compare case-insensitively and trimmed; nested objects compare as subsets.
- `ordered=True` requires the expected tools to occur in the given order.
- `expected_tools: []` means "no tools needed"; with `allow_extra=False`, any call fails.
- A tool that raised is reported as invalid (`tool raised: ...`) even without a schema.

## Argument schemas

`schemas` maps tool name to a JSON-Schema subset: `type` (including lists of types), `enum`,
`required`, `properties`, `additionalProperties: false`, `items`, `minimum`, `maximum`. For
LangChain tools, `tool.get_input_jsonschema()` produces a suitable schema.

`details` lists `expected`, `actual` calls, the per-check `components`, and `invalid` messages.
