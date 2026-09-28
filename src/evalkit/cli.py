"""The ``evalkit`` command line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from evalkit._version import __version__
from evalkit.adapters.base import AgentLoadError, resolve_agent
from evalkit.baseline import compare, format_regressions, load_baseline, save_baseline
from evalkit.capture import TraceExporter
from evalkit.dataset import DatasetError, filter_cases, load_dataset
from evalkit.judge import JudgeError, load_judge
from evalkit.pricing import PricingConfig
from evalkit.registry import DEFAULT_SCORERS, ScorerSpecError, build_scorers
from evalkit.replay import (
    ReplayError,
    StepOverride,
    describe_steps,
    render_diff,
    replay_trace,
)
from evalkit.report import (
    build_report,
    load_report,
    render_html,
    render_terminal,
    write_html,
    write_json,
)
from evalkit.runner import CaseResult, run_eval, score_case
from evalkit.trace import Step, TraceStore

DEFAULT_OUTPUT = "evalkit_results"
EXIT_FAILED = 1
EXIT_USAGE = 2


def _fail(message: str, code: int = EXIT_USAGE) -> click.exceptions.Exit:
    click.secho(f"error: {message}", fg="red", err=True)
    return click.exceptions.Exit(code)


def _text_or_file(value: str | None) -> str | None:
    """``@path`` reads the value from a file."""
    if value is not None and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _parse_thresholds(items: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        try:
            if not sep:
                raise ValueError
            out[name.strip()] = float(raw)
        except ValueError:
            raise click.BadParameter(f"{item!r}: use NAME=VALUE, e.g. accuracy=0.85") from None
    return out


def _check_thresholds(report: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for name, minimum in thresholds.items():
        if name == "pass_rate":
            value: float | None = report["summary"]["pass_rate"]
        elif name in report["scorers"]:
            value = report["scorers"][name]["mean"]
        else:
            failures.append(f"--fail-under {name}: no such scorer in this run")
            continue
        if value is None or value < minimum:
            shown = "n/a" if value is None else f"{value:.3f}"
            failures.append(f"{name} {shown} < required {minimum:.3f}")
    return failures


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="evalkit")
def main() -> None:
    """EvalKit - evaluate and debug LLM agents."""


@main.command()
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Golden set (.jsonl or .csv).",
)
@click.option("--agent", "agent_spec", required=True, help="module:attr or path/to/file.py:attr.")
@click.option(
    "--scorers",
    default=",".join(DEFAULT_SCORERS),
    show_default=True,
    help="Comma-separated built-ins or module:Class custom scorers.",
)
@click.option(
    "--accuracy-mode",
    type=click.Choice(["exact", "fuzzy", "judge"]),
    default="fuzzy",
    show_default=True,
)
@click.option("--judge", "judge_spec", help="LLM judge: 'anthropic[:model]' or 'module:callable'.")
@click.option(
    "--pricing",
    "pricing_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Pricing config (.json/.toml) for cost.",
)
@click.option("--max-latency", type=float, help="Per-request latency budget in seconds.")
@click.option("--max-cost", type=float, help="Per-request cost budget in USD.")
@click.option("--concurrency", "-c", default=4, show_default=True, type=click.IntRange(min=1))
@click.option("--timeout", type=float, help="Per-case timeout in seconds.")
@click.option("--tag", "tags", multiple=True, help="Only run cases with this tag (repeatable).")
@click.option("--limit", type=click.IntRange(min=1), help="Run only the first N cases.")
@click.option(
    "--output-dir",
    "-o",
    default=DEFAULT_OUTPUT,
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option("--otel", is_flag=True, help="Export traces as OpenTelemetry spans.")
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Fail if results regress against this baseline.",
)
@click.option(
    "--save-baseline",
    "save_baseline_path",
    type=click.Path(dir_okay=False),
    help="Write a baseline from this run.",
)
@click.option(
    "--max-regression",
    default=0.02,
    show_default=True,
    type=float,
    help="Allowed absolute drop in a scorer's mean vs the baseline.",
)
@click.option(
    "--max-latency-increase",
    type=float,
    help="Allowed relative p95 latency increase vs baseline (0.25 = +25%).",
)
@click.option(
    "--max-cost-increase",
    type=float,
    help="Allowed relative cost-per-request increase vs baseline.",
)
@click.option(
    "--strict-cases",
    is_flag=True,
    help="Also fail on any case that passed in the baseline but fails now.",
)
@click.option(
    "--fail-under",
    "fail_under",
    multiple=True,
    metavar="NAME=VALUE",
    help="Exit 1 if a scorer's mean (or 'pass_rate') is below VALUE. Repeatable.",
)
@click.option("--quiet", "-q", is_flag=True, help="No per-case progress.")
def run(
    dataset_path: str,
    agent_spec: str,
    scorers: str,
    accuracy_mode: str,
    judge_spec: str | None,
    pricing_path: str | None,
    max_latency: float | None,
    max_cost: float | None,
    concurrency: int,
    timeout: float | None,
    tags: tuple[str, ...],
    limit: int | None,
    output_dir: str,
    otel: bool,
    baseline_path: str | None,
    save_baseline_path: str | None,
    max_regression: float,
    max_latency_increase: float | None,
    max_cost_increase: float | None,
    strict_cases: bool,
    fail_under: tuple[str, ...],
    quiet: bool,
) -> None:
    """Run every case in a dataset through an agent, score it and write reports.

    Exit status: 0 success, 1 a threshold or baseline check failed, 2 usage error.
    """
    thresholds = _parse_thresholds(fail_under)
    try:
        cases = filter_cases(load_dataset(dataset_path), tags=tags, limit=limit)
        adapter = resolve_agent(agent_spec)
        judge = load_judge(judge_spec) if judge_spec else None
        pricing = PricingConfig.load(pricing_path) if pricing_path else None
        scorer_list = build_scorers(
            [s.strip() for s in scorers.split(",") if s.strip()],
            judge=judge,
            accuracy_mode=accuracy_mode,  # type: ignore[arg-type]
            pricing=pricing,
            max_latency_s=max_latency,
            max_cost_usd=max_cost,
        )
        exporters: list[TraceExporter] = []
        if otel:
            from evalkit.otel import OTelExporter

            exporters.append(OTelExporter())
    except (
        DatasetError,
        AgentLoadError,
        JudgeError,
        ScorerSpecError,
        ValueError,
        ImportError,
    ) as exc:
        raise _fail(str(exc)) from exc
    if not cases:
        raise _fail("no cases to run (check --tag/--limit)")

    out = Path(output_dir)
    traces_path = out / "traces.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path.write_text("", encoding="utf-8")

    def progress(done: int, total: int, result: CaseResult) -> None:
        if not quiet:
            mark = (
                click.style("PASS", fg="green") if result.passed else click.style("FAIL", fg="red")
            )
            click.echo(f"[{done}/{total}] {mark} {result.case.id}", err=True)

    results = run_eval(
        cases,
        adapter,
        scorer_list,
        concurrency=concurrency,
        timeout_s=timeout,
        store=TraceStore(traces_path),
        exporters=exporters,
        on_case=progress,
    )
    report = build_report(results)
    write_json(report, out / "report.json")
    write_html(report, out / "report.html")
    click.echo(render_terminal(report, color=sys.stdout.isatty()))
    click.echo(f"\nreport: {out / 'report.html'}\ntraces: {traces_path}")

    if save_baseline_path:
        save_baseline(report, save_baseline_path)
        click.echo(f"baseline written: {save_baseline_path}")

    failures = _check_thresholds(report, thresholds)
    if baseline_path:
        try:
            regressions = compare(
                report,
                load_baseline(baseline_path),
                max_drop=max_regression,
                max_latency_increase=max_latency_increase,
                max_cost_increase=max_cost_increase,
                strict_cases=strict_cases,
            )
        except (ValueError, OSError) as exc:
            raise _fail(str(exc)) from exc
        click.echo("\n" + format_regressions(regressions))
        failures += [r.message for r in regressions]
    if failures:
        click.secho("\nFAILED:\n  " + "\n  ".join(failures), fg="red", err=True)
        raise click.exceptions.Exit(EXIT_FAILED)


@main.command()
@click.argument("trace_id", required=False)
@click.option(
    "--traces",
    "traces_path",
    default=f"{DEFAULT_OUTPUT}/traces.jsonl",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Trace file (JSON lines).",
)
@click.option("--agent", "agent_spec", help="module:attr of the agent to re-run.")
@click.option("--list", "list_traces", is_flag=True, help="List trace ids and exit.")
@click.option("--list-steps", is_flag=True, help="Show the original steps and exit.")
@click.option("--step", type=int, help="Index of the LLM step to modify (see --list-steps).")
@click.option("--model", help="Run the chosen step on this model.")
@click.option("--system", help="Replace the chosen step's system prompt (@file reads a file).")
@click.option("--prompt", help="Replace the chosen step's whole prompt (@file reads a file).")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Ask which step to change and pause after each new step.",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Score the original and new run against this dataset's case.",
)
@click.option("--scorers", default="accuracy,groundedness,safety,tool_use", show_default=True)
@click.option(
    "--save",
    "save_path",
    type=click.Path(dir_okay=False),
    help="Append the replayed trace to this JSONL file.",
)
def replay(
    trace_id: str | None,
    traces_path: str,
    agent_spec: str | None,
    list_traces: bool,
    list_steps: bool,
    step: int | None,
    model: str | None,
    system: str | None,
    prompt: str | None,
    interactive: bool,
    dataset_path: str | None,
    scorers: str,
    save_path: str | None,
) -> None:
    """Re-run a recorded case step by step, optionally swapping the prompt or model.

    Calls before --step are served from the recording; the chosen step runs live with your
    changes; later steps run live. The result is diffed against the original run.
    """
    store = TraceStore(traces_path)
    if list_traces:
        for t in store:
            state = "ERROR" if t.error else "ok"
            click.echo(
                f"{t.trace_id}  {t.case_id or '-':<14} {state:<6} {t.latency_s:6.2f}s  {t.agent}"
            )
        return
    if trace_id is None:
        raise _fail("TRACE_ID is required (use --list to see ids)")
    try:
        original = store.get(trace_id)
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from exc

    click.echo(f"trace {original.trace_id}  case={original.case_id}  agent={original.agent}")
    click.echo(f"input:  {original.input}\nanswer: {original.answer or original.error}\n")
    click.echo(describe_steps(original))
    if list_steps:
        return
    if not agent_spec:
        raise _fail("--agent is required to re-run (or use --list-steps to inspect)")

    if interactive and step is None:
        raw = click.prompt(
            "\nStep to modify (blank = replay unchanged)", default="", show_default=False
        )
        if raw.strip():
            step = int(raw)
            model = click.prompt("New model (blank = keep)", default="", show_default=False) or None
            system = (
                click.prompt("New system prompt (blank = keep)", default="", show_default=False)
                or None
            )
            prompt = (
                click.prompt("New full prompt (blank = keep)", default="", show_default=False)
                or None
            )
    try:
        override = StepOverride(
            model=model, system=_text_or_file(system), prompt=_text_or_file(prompt)
        )
        adapter = resolve_agent(agent_spec)
    except (AgentLoadError, OSError) as exc:
        raise _fail(str(exc)) from exc

    click.echo(
        "\nreplaying"
        + (
            f" with changes at step {step}: {override.describe()}"
            if step is not None
            else " from the recording"
        )
        + "\n"
    )

    def on_step(s: Step) -> None:
        tag = " (cached)" if s.llm is not None and s.llm.cached else ""
        click.echo(f"  > {s.kind:<9} {s.name:<28} {s.latency_s * 1000:8.1f} ms{tag}")
        if interactive:
            click.pause("    press any key for the next step...")

    try:
        result = replay_trace(
            original,
            adapter,
            step=step,
            override=None if override.is_empty else override,
            on_step=on_step,
            store=TraceStore(save_path) if save_path else None,
        )
    except ReplayError as exc:
        raise _fail(str(exc)) from exc

    click.echo("\n" + render_diff(result.diff, color=sys.stdout.isatty()))
    click.echo(f"\nnew trace id: {result.replayed.trace_id}")
    if result.replayed.error:
        click.secho(f"replayed run failed: {result.replayed.error}", fg="red", err=True)

    if dataset_path:
        cases = {c.id: c for c in load_dataset(dataset_path)}
        case = cases.get(original.case_id or "")
        if case is None:
            click.echo(f"\n(case {original.case_id!r} not in dataset; skipping scoring)")
            return
        scorer_list = build_scorers([s.strip() for s in scorers.split(",") if s.strip()])
        before = {s.scorer: s for s in score_case(case, original, scorer_list)}
        after = {s.scorer: s for s in score_case(case, result.replayed, scorer_list)}
        click.echo("\nscores (original -> replayed)")
        for name, b in before.items():
            a = after[name]
            click.echo(f"  {name:<14} {b.score:.2f} -> {a.score:.2f}   {a.reason[:90]}")


@main.command()
@click.option(
    "--traces",
    "traces_path",
    default=f"{DEFAULT_OUTPUT}/traces.jsonl",
    show_default=True,
    type=click.Path(dir_okay=False),
)
@click.option("--failed", is_flag=True, help="Only traces whose agent run raised an error.")
def traces(traces_path: str, failed: bool) -> None:
    """List recorded traces (id, case, status, latency)."""
    for t in TraceStore(traces_path):
        if failed and not t.error:
            continue
        click.echo(
            f"{t.trace_id}  {t.case_id or '-':<14} {'ERROR' if t.error else 'ok':<6} "
            f"{t.latency_s:6.2f}s  {len(t.steps)} steps  tokens={t.usage.total_tokens}"
        )


@main.command()
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--html", "html_path", type=click.Path(dir_okay=False), help="Write the HTML report here."
)
def report(report_path: str, html_path: str | None) -> None:
    """Print a report.json summary, optionally re-rendering its HTML."""
    try:
        data = load_report(report_path)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    click.echo(render_terminal(data, color=sys.stdout.isatty()))
    if html_path:
        Path(html_path).write_text(render_html(data), encoding="utf-8")
        click.echo(f"\nhtml: {html_path}")


@main.command("baseline")
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False))
def baseline_cmd(report_path: str, output: str) -> None:
    """Create a baseline file from a report.json (commit it to track regressions)."""
    try:
        save_baseline(load_report(report_path), output)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    click.echo(f"baseline written: {output}")


@main.command("compare")
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--max-regression", default=0.02, show_default=True, type=float)
@click.option("--max-latency-increase", type=float)
@click.option("--max-cost-increase", type=float)
@click.option("--strict-cases", is_flag=True)
def compare_cmd(
    baseline_path: str,
    report_path: str,
    max_regression: float,
    max_latency_increase: float | None,
    max_cost_increase: float | None,
    strict_cases: bool,
) -> None:
    """Compare a report to a baseline; exit 1 on regressions."""
    try:
        regressions = compare(
            load_report(report_path),
            load_baseline(baseline_path),
            max_drop=max_regression,
            max_latency_increase=max_latency_increase,
            max_cost_increase=max_cost_increase,
            strict_cases=strict_cases,
        )
    except (ValueError, OSError) as exc:
        raise _fail(str(exc)) from exc
    click.echo(format_regressions(regressions))
    if regressions:
        raise click.exceptions.Exit(EXIT_FAILED)


if __name__ == "__main__":  # pragma: no cover
    main()
