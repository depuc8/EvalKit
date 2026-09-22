"""Reports: a JSON document, a terminal summary and a static HTML page with drill-down."""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any

import click

from evalkit._version import __version__
from evalkit.runner import EvalResults

Report = dict[str, Any]

METRIC_LABELS: dict[str, str] = {
    "latency_p50_s": "latency p50 (s)",
    "latency_p95_s": "latency p95 (s)",
    "ttft_p50_s": "time to first token p50 (s)",
    "ttft_p95_s": "time to first token p95 (s)",
    "tokens_per_second_mean": "tokens / second (mean)",
    "input_tokens_mean": "input tokens (mean)",
    "output_tokens_mean": "output tokens (mean)",
    "cost_per_request_usd": "cost per request (USD)",
    "total_cost_usd": "total cost (USD)",
}


def build_report(results: EvalResults, *, include_traces: bool = True) -> Report:
    """Convert results into a JSON-serialisable report."""
    cases: list[dict[str, Any]] = []
    for c in results.cases:
        item: dict[str, Any] = {
            "id": c.case.id,
            "input": c.case.input,
            "reference": c.case.reference,
            "tags": c.case.tags,
            "passed": c.passed,
            "error": c.error,
            "trace_id": c.trace.trace_id,
            "latency_s": c.trace.latency_s,
            "answer": c.trace.answer,
            "scores": [s.model_dump(mode="json") for s in c.scores],
        }
        if include_traces:
            item["trace"] = c.trace.model_dump(mode="json")
        cases.append(item)
    failed = sum(not c.passed for c in results.cases)
    return {
        "evalkit_version": __version__,
        "run_id": results.run_id,
        "agent": results.agent,
        "created_at": dt.datetime.fromtimestamp(results.started_at, dt.timezone.utc).isoformat(),
        "duration_s": results.duration_s,
        "summary": {
            "cases": len(results.cases),
            "passed": len(results.cases) - failed,
            "failed": failed,
            "errors": sum(c.error is not None for c in results.cases),
            "pass_rate": results.pass_rate,
        },
        "scorers": {k: v.model_dump(mode="json") for k, v in results.summary().items()},
        "cases": cases,
    }


def write_json(report: Report, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_report(path: str | Path) -> Report:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "scorers" not in data or "cases" not in data:
        raise ValueError(f"{path} is not an EvalKit report")
    return data


# -- terminal -----------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_terminal(report: Report, *, color: bool = False, max_failures: int = 10) -> str:
    def paint(text: str, fg: str | None = None, bold: bool = False) -> str:
        return click.style(text, fg=fg, bold=bold) if color else text

    s = report["summary"]
    lines = [
        paint(f"EvalKit - {report['agent']}", bold=True),
        f"{s['cases']} cases in {report['duration_s']:.1f}s   "
        + paint(f"{s['passed']} passed", "green")
        + "   "
        + paint(f"{s['failed']} failed", "red" if s["failed"] else None)
        + (f"   ({s['errors']} agent errors)" if s["errors"] else ""),
        "",
        f"{'scorer':<16}{'mean':>8}{'pass rate':>11}{'scored':>8}{'skipped':>9}",
        "-" * 52,
    ]
    for name, sc in report["scorers"].items():
        row = (
            f"{name:<16}{_fmt(sc['mean']):>8}{_pct(sc['pass_rate']):>11}"
            f"{sc['n']:>8}{sc['skipped']:>9}"
        )
        lines.append(row)
    metrics = {k: v for sc in report["scorers"].values() for k, v in sc.get("metrics", {}).items()}
    if metrics:
        lines += ["", "performance"]
        lines += [f"  {METRIC_LABELS.get(k, k):<32}{v:>12.4g}" for k, v in metrics.items()]
    failures = [c for c in report["cases"] if not c["passed"]]
    if failures:
        lines += ["", paint(f"failures ({len(failures)})", "red", bold=True)]
        for case in failures[:max_failures]:
            lines.append(f"  {case['id']}  [trace {case['trace_id'][:8]}]")
            if case["error"]:
                lines.append(f"      agent error: {case['error']}")
            for sc in case["scores"]:
                if not sc["passed"] and not sc["skipped"]:
                    lines.append(f"      {sc['scorer']} {sc['score']:.2f}: {sc['reason'][:110]}")
        if len(failures) > max_failures:
            lines.append(f"  ... and {len(failures) - max_failures} more (see the HTML report)")
    return "\n".join(lines)


# -- HTML ---------------------------------------------------------------------------------
_CSS = """
:root{--bg:#f6f7f9;--fg:#1c2330;--muted:#667085;--card:#fff;--line:#e3e6eb;--ok:#18794e;
--okbg:#e6f6ee;--bad:#b42318;--badbg:#fdecea;--warn:#a15c07;--warnbg:#fff4e0;--accent:#3b5bdb}
@media(prefers-color-scheme:dark){:root{--bg:#12151b;--fg:#e6e9ef;--muted:#9aa3b2;--card:#1b2029;
--line:#2c3340;--ok:#5fd29a;--okbg:#14301f;--bad:#ff8b80;--badbg:#38191a;--warn:#f2b45a;
--warnbg:#3a2a10;--accent:#8ea2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.muted{color:var(--muted)}.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;
min-width:130px}.card b{display:block;font-size:24px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600}
.pass{background:var(--okbg);color:var(--ok)}.fail{background:var(--badbg);color:var(--bad)}
.skip{background:var(--warnbg);color:var(--warn)}
details.case{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0}
details.case>summary{cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:center;
flex-wrap:wrap}details.case[open]>summary{border-bottom:1px solid var(--line)}
.body{padding:12px 16px}.body h3{font-size:13px;text-transform:uppercase;color:var(--muted);
letter-spacing:.04em;margin:16px 0 6px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;
overflow:auto;white-space:pre-wrap;word-break:break-word;margin:4px 0;font-size:13px}
.bar{height:6px;background:var(--accent);border-radius:3px;min-width:2px}
label.filter{display:inline-flex;gap:6px;align-items:center;margin:8px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
@media(max-width:600px){th,td{padding:6px 8px}}
"""

_JS = """
const box=document.getElementById('only-fail');
if(box){box.addEventListener('change',()=>{document.querySelectorAll('details.case').forEach(
d=>{d.hidden=box.checked&&d.dataset.passed==='1'})})}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _pre(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    return f"<pre>{_e(text)}</pre>"


def _pill(ok: bool, skipped: bool = False) -> str:
    if skipped:
        return '<span class="pill skip">skipped</span>'
    return f'<span class="pill {"pass" if ok else "fail"}">{"pass" if ok else "fail"}</span>'


def _score_details(score: dict[str, Any]) -> str:
    details = score.get("details") or {}
    unsupported = details.get("unsupported_claims")
    findings = details.get("findings")
    out = ""
    if unsupported:
        out += (
            "<div><b>Unsupported claims</b><ul>"
            + "".join(f"<li>{_e(c)}</li>" for c in unsupported)
            + "</ul></div>"
        )
    if findings:
        out += (
            "<div><b>Findings</b><ul>"
            + "".join(
                f"<li>{_e(f['category'])}/{_e(f['rule'])} ({_e(f['severity'])}): "
                f"<code>{_e(f['match'])}"
                "</code></li>"
                for f in findings
            )
            + "</ul></div>"
        )
    extra = {
        k: v for k, v in details.items() if k not in ("unsupported_claims", "findings", "claims")
    }
    if extra:
        out += f"<details><summary class='muted'>details</summary>{_pre(extra)}</details>"
    return out


def _steps_html(trace: dict[str, Any]) -> str:
    steps = trace.get("steps", [])
    if not steps:
        return '<p class="muted">no steps recorded</p>'
    peak = max((s["latency_s"] for s in steps), default=0) or 1.0
    rows = []
    for i, st in enumerate(steps):
        body = ""
        llm, tool = st.get("llm"), st.get("tool")
        if llm:
            prompt = "\n".join(f"[{m['role']}] {m['content']}" for m in llm["prompt"])
            usage = llm["usage"]
            body += (
                f"<div class='muted'>model {_e(llm['model'])} · {usage['input_tokens']} in / "
                f"{usage['output_tokens']} out tokens"
                + (
                    f" · ttft {llm['time_to_first_token_s']:.3f}s"
                    if llm.get("time_to_first_token_s") is not None
                    else ""
                )
                + (" · served from replay cache" if llm.get("cached") else "")
                + f"</div><b>Prompt</b>{_pre(prompt)}<b>Output</b>{_pre(llm['output'])}"
            )
            if llm.get("tool_calls"):
                body += f"<b>Requested tool calls</b>{_pre(llm['tool_calls'])}"
        elif tool:
            body += f"<b>Arguments</b>{_pre(tool['args'])}"
            body += (
                f"<b>Result</b>{_pre(tool['result'] if tool['error'] is None else tool['error'])}"
            )
        else:
            if st.get("inputs") is not None:
                body += f"<b>Inputs</b>{_pre(st['inputs'])}"
            if st.get("outputs") is not None:
                body += f"<b>Outputs</b>{_pre(st['outputs'])}"
        if st.get("documents"):
            body += "<b>Documents</b>" + "".join(_pre(d) for d in st["documents"])
        if st.get("error"):
            body += f"<b>Error</b>{_pre(st['error'])}"
        width = max(2, int(st["latency_s"] / peak * 100))
        rows.append(
            f"<tr><td class='n'>{i}</td><td>{_e(st['kind'])}</td>"
            f"<td><details><summary>{_e(st['name'])}</summary>{body}</details></td>"
            f"<td class='n'>{st['latency_s'] * 1000:.1f} ms</td>"
            f"<td style='width:20%'><div class='bar' style='width:{width}%'></div></td></tr>"
        )
    return (
        "<table><tr><th class='n'>#</th><th>kind</th><th>step</th><th class='n'>latency</th>"
        "<th></th></tr>" + "".join(rows) + "</table>"
    )


def _case_html(case: dict[str, Any]) -> str:
    chips = "".join(
        f"<span class='pill {'skip' if s['skipped'] else ('pass' if s['passed'] else 'fail')}'>"
        f"{_e(s['scorer'])} {s['score']:.2f}</span>"
        for s in case["scores"]
    )
    score_rows = "".join(
        f"<tr><td>{_e(s['scorer'])}</td><td class='n'>{s['score']:.2f}</td>"
        f"<td>{_pill(s['passed'], s['skipped'])}</td>"
        f"<td>{_e(s['reason'])}{_score_details(s)}</td></tr>"
        for s in case["scores"]
    )
    parts = [
        "<h3>Input</h3>",
        _pre(case["input"]),
        "<h3>Answer</h3>",
        _pre(case["answer"] or "(no answer)"),
    ]
    if case.get("reference") is not None:
        parts += ["<h3>Reference</h3>", _pre(case["reference"])]
    if case.get("error"):
        parts += ["<h3>Agent error</h3>", _pre(case["error"])]
    parts += [
        "<h3>Scores</h3><table><tr><th>scorer</th><th class='n'>score</th><th>result</th>"
        f"<th>reason</th></tr>{score_rows}</table>",
    ]
    if "trace" in case:
        parts += [f"<h3>Trace <code>{_e(case['trace_id'])}</code></h3>", _steps_html(case["trace"])]
    return (
        f"<details class='case' data-passed='{1 if case['passed'] else 0}'><summary>"
        f"{_pill(case['passed'])} <b>{_e(case['id'])}</b> {chips}"
        f"<span class='muted'>{case['latency_s']:.2f}s</span></summary>"
        f"<div class='body'>{''.join(parts)}</div></details>"
    )


def render_html(report: Report) -> str:
    """Render a self-contained static HTML report (no external assets)."""
    s = report["summary"]
    cards = "".join(
        f"<div class='card'>{label}<b>{value}</b></div>"
        for label, value in (
            ("cases", s["cases"]),
            ("passed", s["passed"]),
            ("failed", s["failed"]),
            ("pass rate", f"{s['pass_rate'] * 100:.0f}%"),
            ("duration", f"{report['duration_s']:.1f}s"),
        )
    )
    scorer_rows = "".join(
        f"<tr><td>{_e(name)}</td><td class='n'>{_fmt(sc['mean'])}</td>"
        f"<td class='n'>{_pct(sc['pass_rate'])}</td><td class='n'>{sc['n']}</td>"
        f"<td class='n'>{sc['skipped']}</td></tr>"
        for name, sc in report["scorers"].items()
    )
    metrics = {k: v for sc in report["scorers"].values() for k, v in sc.get("metrics", {}).items()}
    metric_html = ""
    if metrics:
        metric_html = (
            "<h2>Performance</h2><table>"
            + "".join(
                f"<tr><td>{_e(METRIC_LABELS.get(k, k))}</td><td class='n'>{v:.4g}</td></tr>"
                for k, v in metrics.items()
            )
            + "</table>"
        )
    cases = "".join(_case_html(c) for c in report["cases"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvalKit report - {_e(report["agent"])}</title><style>{_CSS}</style></head>
<body><main>
<h1>EvalKit report</h1>
<div class="muted">agent <code>{_e(report["agent"])}</code> ·
 run <code>{_e(report["run_id"][:8])}</code>
 · {_e(report["created_at"])} · evalkit {_e(report["evalkit_version"])}</div>
<div class="cards">{cards}</div>
<h2>Scorers</h2>
<table><tr><th>scorer</th><th class="n">mean</th><th class="n">pass rate</th>
<th class="n">scored</th>
<th class="n">skipped</th></tr>{scorer_rows}</table>
{metric_html}
<h2>Cases</h2>
<label class="filter"><input type="checkbox" id="only-fail"> show failures only</label>
{cases}
</main><script>{_JS}</script></body></html>
"""


def write_html(report: Report, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(report), encoding="utf-8")
    return p
