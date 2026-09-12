"""Tool-use correctness: right tools, right arguments, valid calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from evalkit.dataset import Case, ExpectedTool
from evalkit.scorers.base import Score, Scorer
from evalkit.trace import ToolCall, Trace


def _type_ok(name: str, value: Any) -> bool:
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list | tuple)
    if name == "object":
        return isinstance(value, dict)
    if name == "null":
        return value is None
    return True  # unknown type keyword: do not fail the call


def validate_args(schema: Mapping[str, Any], value: Any, path: str = "args") -> list[str]:
    """Validate ``value`` against a JSON-Schema subset; returns human-readable errors.

    Supports ``type``, ``enum``, ``required``, ``properties``, ``additionalProperties: false``,
    ``items``, ``minimum`` and ``maximum`` - enough for typical tool argument schemas.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_type_ok(n, value) for n in names):
            return [f"{path}: expected {'/'.join(names)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {list(schema['enum'])}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        props: Mapping[str, Any] = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required argument {key!r}")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate_args(props[key], sub, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected argument {key!r}")
    if isinstance(value, list | tuple) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate_args(schema["items"], item, f"{path}[{i}]"))
    return errors


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _same(v, actual[k]) for k, v in expected.items())
    return bool(expected == actual)


def args_match(expected: ExpectedTool, call: ToolCall) -> bool:
    """``expected.args`` must be a subset of the call's arguments (strings compared loosely)."""
    return expected.args is None or _same(expected.args, call.args)


class ToolUseScorer(Scorer):
    """Did the agent call the expected tools with valid arguments?

    Checks (the score is the mean of those that apply):

    * **match** - fraction of ``case.expected_tools`` found among the calls (name, plus the
      expected argument subset when given). With ``ordered=True`` they must appear in order.
    * **valid** - fraction of calls whose arguments satisfy ``schemas[tool_name]`` and
      that did not raise an error (only when schemas are supplied or a call errored).
    * **no extras** - with ``allow_extra=False``, calls to tools that were not expected
      reduce the score.

    An empty ``expected_tools`` list means "no tools should be needed".
    """

    default_name: ClassVar[str] = "tool_use"
    default_threshold: ClassVar[float] = 1.0

    def __init__(
        self,
        *,
        schemas: Mapping[str, Mapping[str, Any]] | None = None,
        ordered: bool = False,
        allow_extra: bool = True,
        name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.schemas = dict(schemas or {})
        self.ordered = ordered
        self.allow_extra = allow_extra

    def _match(
        self, expected: Sequence[ExpectedTool], calls: Sequence[ToolCall]
    ) -> tuple[list[ExpectedTool], list[ToolCall]]:
        """Return (missing expected tools, calls matched to an expectation)."""
        used: set[int] = set()
        missing: list[ExpectedTool] = []
        matched: list[ToolCall] = []
        cursor = 0
        for exp in expected:
            start = cursor if self.ordered else 0
            hit = next(
                (
                    i
                    for i in range(start, len(calls))
                    if i not in used and calls[i].name == exp.name and args_match(exp, calls[i])
                ),
                None,
            )
            if hit is None:
                missing.append(exp)
                continue
            used.add(hit)
            matched.append(calls[hit])
            if self.ordered:
                cursor = hit + 1
        return missing, matched

    def score(self, case: Case, trace: Trace) -> Score:
        if case.expected_tools is None:
            return self.skip("case declares no expected tools")
        expected, calls = case.expected_tools, trace.tool_calls
        missing, matched = self._match(expected, calls)
        parts: dict[str, float] = {}
        notes: list[str] = []

        if expected:
            parts["match"] = (len(expected) - len(missing)) / len(expected)
            if missing:
                notes.append("missing: " + ", ".join(self._describe(m) for m in missing))
        elif calls and not self.allow_extra:
            parts["match"] = 0.0

        invalid: list[str] = []
        checked = bad_calls = 0
        for call in calls:
            schema = self.schemas.get(call.name)
            if schema is None and call.error is None:
                continue
            checked += 1
            problems = validate_args(schema, call.args) if schema is not None else []
            if call.error:
                problems.append(f"tool raised: {call.error}")
            if problems:
                bad_calls += 1
                invalid.extend(f"{call.name}: {p}" for p in problems)
        if checked:
            parts["valid"] = 1 - bad_calls / checked
            notes.extend(f"invalid call - {p}" for p in invalid)

        wanted = {e.name for e in expected}
        extras = [c.name for c in calls if c not in matched and c.name not in wanted]
        if not self.allow_extra and calls:
            parts["no_extras"] = 1 - len(extras) / len(calls)
            if extras:
                notes.append("unexpected tools: " + ", ".join(extras))

        if not parts:
            parts["match"] = 1.0
        value = sum(parts.values()) / len(parts)
        reason = "tool use correct" if value >= 1.0 else "; ".join(notes) or "tool use incorrect"
        return self.result(
            value,
            reason,
            expected=[e.model_dump() for e in expected],
            actual=[{"name": c.name, "args": c.args, "error": c.error} for c in calls],
            components=parts,
            invalid=invalid,
        )

    @staticmethod
    def _describe(tool: ExpectedTool) -> str:
        return tool.name if tool.args is None else f"{tool.name}({tool.args})"
