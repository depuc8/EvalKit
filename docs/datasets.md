# Datasets

A dataset is a list of **cases**, loaded from JSONL (one object per line) or CSV.

| Field | Required | Meaning |
| --- | --- | --- |
| `input` | yes | What is sent to the agent. A string, or an object passed through as the graph input |
| `id` | no | Unique case id; defaults to `case-001`, `case-002`, ... |
| `reference` | no | Reference answer for the accuracy scorer. Aliases: `reference_answer`, `expected_output`, `expected`, `answer` |
| `expected_tools` | no | Tools the agent should call: names, or `{"name": ..., "args": {...}}` (see [tool use](scorers/tool-use.md)) |
| `context` | no | Golden context for the groundedness scorer. A string or list of strings |
| `tags` | no | Used with `evalkit run --tag` |
| `metadata` | no | Free-form; `canary` and `allowed_pii` are read by the [safety](scorers/safety.md) scorer |

## JSONL

```json
{"id": "pro-price", "input": "How much does Pro cost?", "reference": "The Pro plan costs $49 per user per month.", "expected_tools": ["search_kb"], "context": ["The Pro plan costs $49 per user per month, billed monthly."], "tags": ["lookup"]}
```

Blank lines and lines starting with `#` are ignored.

## CSV

Header row required. List columns accept JSON, or a simple delimited form:

| Column | Simple form | JSON form |
| --- | --- | --- |
| `expected_tools`, `tags` | `search, calculator` | `[{"name": "calculator", "args": {"expression": "1+1"}}]` |
| `context` | `passage one\|passage two` | `["passage one", "passage two"]` |
| `metadata` | - | `{"canary": "PWNED"}` |

Empty cells are treated as missing.

## Errors

Malformed files fail fast with a location, e.g. `golden.jsonl:17: missing required field 'input'`
or `golden.jsonl:3: duplicate case id 'refund-1'`.

```python
from evalkit import load_dataset
cases = load_dataset("golden.jsonl")     # .jsonl / .ndjson / .csv
```
