"""Golden datasets: JSONL and CSV loaders."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

_REFERENCE_KEYS = ("reference", "reference_answer", "expected_output", "expected", "answer")


class DatasetError(ValueError):
    """The dataset file is malformed."""


class ExpectedTool(BaseModel):
    """A tool the agent is expected to call, optionally with (a subset of) its arguments."""

    name: str
    args: dict[str, Any] | None = None


class Case(BaseModel):
    """One golden test case."""

    id: str
    input: Any
    reference: str | None = None
    expected_tools: list[ExpectedTool] | None = None
    context: list[str] | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_tools", mode="before")
    @classmethod
    def _coerce_tools(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        return [{"name": v} if isinstance(v, str) else v for v in value]

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @property
    def input_text(self) -> str:
        if isinstance(self.input, str):
            return self.input
        return json.dumps(self.input, ensure_ascii=False)


def _normalize_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    record = dict(raw)
    for key in _REFERENCE_KEYS[1:]:
        if key in record and "reference" not in record:
            record["reference"] = record.pop(key)
    if record.get("id") in (None, ""):
        record["id"] = f"case-{index:03d}"
    record["id"] = str(record["id"])
    return record


def _build(records: Iterable[tuple[int, dict[str, Any]]], source: str) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for index, raw in records:
        if "input" not in raw:
            raise DatasetError(f"{source}:{index}: missing required field 'input'")
        try:
            case = Case.model_validate(_normalize_record(raw, len(cases) + 1))
        except ValidationError as exc:
            err = exc.errors()[0]
            where = ".".join(str(part) for part in err["loc"])
            raise DatasetError(f"{source}:{index}: {where}: {err['msg']}") from exc
        if case.id in seen:
            raise DatasetError(f"{source}:{index}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def load_jsonl(path: str | Path) -> list[Case]:
    p = Path(path)
    records: list[tuple[int, dict[str, Any]]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"{p}:{lineno}: each line must be a JSON object")
        records.append((lineno, value))
    return _build(records, str(p))


def _split_list(value: str, sep: str) -> list[str]:
    return [part.strip() for part in value.split(sep) if part.strip()]


def _csv_value(key: str, value: str) -> Any:
    if value.strip() == "":
        return None
    stripped = value.strip()
    if key in ("expected_tools", "context", "tags", "metadata") and stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"column {key!r}: invalid JSON ({exc.msg})") from exc
    if key in ("expected_tools", "tags"):
        return _split_list(stripped, ",")
    if key == "context":
        return _split_list(stripped, "|")
    return value


def load_csv(path: str | Path) -> list[Case]:
    p = Path(path)
    records: list[tuple[int, dict[str, Any]]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record: dict[str, Any] = {}
            for key, raw in row.items():
                if key is None or raw is None:
                    continue
                value = _csv_value(key, raw)
                if value is not None:
                    record[key] = value
            records.append((reader.line_num, record))
    return _build(records, str(p))


def load_dataset(path: str | Path) -> list[Case]:
    """Load cases from ``.jsonl``/``.ndjson`` or ``.csv`` (chosen by extension)."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return load_jsonl(p)
    if suffix == ".csv":
        return load_csv(p)
    raise DatasetError(f"unsupported dataset format {suffix!r} (use .jsonl or .csv)")


def filter_cases(
    cases: Sequence[Case], *, tags: Sequence[str] = (), limit: int | None = None
) -> list[Case]:
    selected = [c for c in cases if not tags or set(tags) & set(c.tags)]
    return selected[:limit] if limit is not None else selected
