"""Baselines and regression detection for CI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.report import Report

BASELINE_VERSION = 1
#: Run-level metrics (from scorer ``aggregate``) tracked in the baseline.
TRACKED_METRICS = ("latency_p95_s", "cost_per_request_usd")


@dataclass(frozen=True)
class Regression:
    kind: str  # "scorer", "metric", "case" or "missing"
    name: str
    baseline: float | None
    current: float | None
    message: str


def make_baseline(report: Report) -> dict[str, Any]:
    """Distil a report into a small, diff-friendly baseline."""
    metrics: dict[str, dict[str, float]] = {}
    for scorer, sc in report["scorers"].items():
        tracked = {k: v for k, v in sc.get("metrics", {}).items() if k in TRACKED_METRICS}
        if tracked:
            metrics[scorer] = tracked
    return {
        "version": BASELINE_VERSION,
        "agent": report["agent"],
        "scorers": {
            name: {"mean": sc["mean"], "pass_rate": sc["pass_rate"]}
            for name, sc in report["scorers"].items()
        },
        "metrics": metrics,
        "cases": {
            c["id"]: {
                "passed": c["passed"],
                "scores": {s["scorer"]: s["score"] for s in c["scores"] if not s["skipped"]},
            }
            for c in report["cases"]
        },
    }


def save_baseline(report: Report, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(make_baseline(report), indent=2, sort_keys=True) + "\n", "utf-8")
    return p


def load_baseline(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != BASELINE_VERSION:
        raise ValueError(f"{path} is not a version-{BASELINE_VERSION} EvalKit baseline")
    return data


def compare(
    report: Report,
    baseline: dict[str, Any],
    *,
    max_drop: float = 0.02,
    max_latency_increase: float | None = None,
    max_cost_increase: float | None = None,
    strict_cases: bool = False,
) -> list[Regression]:
    """Return regressions of ``report`` against ``baseline`` (empty list = OK).

    Args:
        max_drop: tolerated absolute drop in each scorer's mean score.
        max_latency_increase / max_cost_increase: tolerated relative increase (0.25 = +25%) of
            p95 latency / cost per request; ``None`` disables the check (latency is noisy in CI).
        strict_cases: also flag every case that passed in the baseline but fails now.
    """
    found: list[Regression] = []
    for name, base in baseline["scorers"].items():
        cur = report["scorers"].get(name)
        if cur is None:
            found.append(
                Regression(
                    "missing",
                    name,
                    base["mean"],
                    None,
                    f"scorer {name!r} is in the baseline but not in this run",
                )
            )
            continue
        if base["mean"] is None or cur["mean"] is None:
            continue
        drop = base["mean"] - cur["mean"]
        if drop > max_drop + 1e-9:
            found.append(
                Regression(
                    "scorer",
                    name,
                    base["mean"],
                    cur["mean"],
                    f"{name} mean fell {base['mean']:.3f} -> {cur['mean']:.3f} "
                    f"(-{drop:.3f}, allowed -{max_drop:.3f})",
                )
            )
    limits = {"latency_p95_s": max_latency_increase, "cost_per_request_usd": max_cost_increase}
    for scorer, tracked in baseline.get("metrics", {}).items():
        current = report["scorers"].get(scorer, {}).get("metrics", {})
        for key, base_value in tracked.items():
            limit = limits.get(key)
            if limit is None or key not in current or base_value <= 0:
                continue
            rise = current[key] / base_value - 1
            if rise > limit + 1e-9:
                found.append(
                    Regression(
                        "metric",
                        f"{scorer}.{key}",
                        base_value,
                        current[key],
                        f"{key} rose {base_value:.4g} -> {current[key]:.4g} "
                        f"(+{rise * 100:.0f}%, allowed +{limit * 100:.0f}%)",
                    )
                )
    if strict_cases:
        now = {c["id"]: c for c in report["cases"]}
        for case_id, base_case in baseline.get("cases", {}).items():
            cur_case = now.get(case_id)
            if base_case["passed"] and cur_case is not None and not cur_case["passed"]:
                found.append(
                    Regression(
                        "case",
                        case_id,
                        1.0,
                        0.0,
                        f"case {case_id} passed in the baseline but fails now",
                    )
                )
    return found


def format_regressions(regressions: list[Regression]) -> str:
    if not regressions:
        return "No regressions against baseline."
    return "Regressions against baseline:\n" + "\n".join(f"  - {r.message}" for r in regressions)
