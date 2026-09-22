"""Assertion helper for tests and CI: ``assert_eval(results, min_accuracy=0.85)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalkit.baseline import compare, load_baseline
from evalkit.report import Report, build_report
from evalkit.runner import EvalResults


class EvalAssertionError(AssertionError):
    """Raised by :func:`assert_eval` when a threshold is violated."""


def assert_eval(
    results: EvalResults | Report,
    *,
    min_pass_rate: float | None = None,
    max_latency_p95_s: float | None = None,
    max_cost_per_request_usd: float | None = None,
    baseline: str | Path | dict[str, Any] | None = None,
    max_drop: float = 0.02,
    max_latency_increase: float | None = None,
    max_cost_increase: float | None = None,
    **min_scores: float,
) -> None:
    """Fail (``AssertionError``) unless the run meets every threshold.

    ``min_<scorer>=x`` requires that scorer's mean score to be at least ``x`` - for example
    ``min_accuracy=0.85`` or ``min_groundedness=0.9`` (custom scorers work too, by name).
    ``baseline`` compares against a saved baseline (see ``evalkit baseline``) and fails on
    regressions larger than ``max_drop``.
    """
    report = results if isinstance(results, dict) else build_report(results, include_traces=False)
    scorers: dict[str, Any] = report["scorers"]
    failures: list[str] = []

    for key, minimum in min_scores.items():
        if not key.startswith("min_"):
            raise TypeError(f"assert_eval() got an unexpected argument {key!r}")
        name = key[len("min_") :]
        if name not in scorers:
            raise ValueError(f"no scorer named {name!r} in results (have: {sorted(scorers)})")
        mean = scorers[name]["mean"]
        if mean is None:
            failures.append(f"{name}: no cases were scored")
        elif mean < minimum:
            failures.append(f"{name} mean {mean:.3f} < required {minimum:.3f}")

    if min_pass_rate is not None and report["summary"]["pass_rate"] < min_pass_rate:
        failures.append(
            f"pass rate {report['summary']['pass_rate']:.3f} < required {min_pass_rate:.3f}"
        )

    metrics = {k: v for sc in scorers.values() for k, v in sc.get("metrics", {}).items()}
    for label, limit in (
        ("latency_p95_s", max_latency_p95_s),
        ("cost_per_request_usd", max_cost_per_request_usd),
    ):
        if limit is None:
            continue
        if label not in metrics:
            failures.append(f"{label} was not measured (is the latency_cost scorer enabled?)")
        elif metrics[label] > limit:
            failures.append(f"{label} {metrics[label]:.4g} > allowed {limit:.4g}")

    if baseline is not None:
        data = baseline if isinstance(baseline, dict) else load_baseline(baseline)
        failures += [
            r.message
            for r in compare(
                report,
                data,
                max_drop=max_drop,
                max_latency_increase=max_latency_increase,
                max_cost_increase=max_cost_increase,
            )
        ]

    if failures:
        raise EvalAssertionError("EvalKit assertion failed:\n  - " + "\n  - ".join(failures))
