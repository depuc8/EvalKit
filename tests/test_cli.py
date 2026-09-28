from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from evalkit.cli import main
from evalkit.trace import TraceStore

AGENT_SRC = """
from evalkit import current_recorder

def agent(question):
    rec = current_recorder()
    rec.record_tool("search", {"q": question}, result="doc")
    rec.record_llm(model="m", prompt=question, output="answer", input_tokens=5, output_tokens=2)
    return "yes" if question == "good" else "no"
"""

BASELINE_CASES = [
    {"id": "g", "input": "good", "reference": "yes", "expected_tools": ["search"]},
    {"id": "b", "input": "bad", "reference": "yes", "expected_tools": ["search"], "tags": ["hard"]},
]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "myagent.py").write_text(AGENT_SRC)
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(c) for c in BASELINE_CASES))
    (tmp_path / "pricing.json").write_text(
        json.dumps({"default": {"input_per_million": 1, "output_per_million": 1}})
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def invoke(*args: str) -> Result:
    return CliRunner().invoke(main, list(args))


def run_args(*extra: str) -> list[str]:
    return ["run", "--dataset", "golden.jsonl", "--agent", "myagent.py:agent", "-q", *extra]


def test_version_and_help() -> None:
    assert "evalkit" in invoke("--version").output
    out = invoke("--help").output
    for command in ("run", "replay", "report", "baseline", "compare", "traces"):
        assert command in out


def test_run_writes_reports(workspace: Path) -> None:
    result = invoke(
        *run_args("--pricing", "pricing.json", "--scorers", "accuracy,tool_use,latency_cost")
    )
    assert result.exit_code == 0, result.output
    assert "2 cases" in result.output and "accuracy" in result.output
    out = workspace / "evalkit_results"
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["passed"] == 1
    assert "<details" in (out / "report.html").read_text()
    assert len(TraceStore(out / "traces.jsonl").read_all()) == 2
    assert "cost_per_request_usd" in report["scorers"]["latency_cost"]["metrics"]


def test_run_filters_and_options(workspace: Path) -> None:
    result = invoke(
        *run_args(
            "--tag",
            "hard",
            "--limit",
            "5",
            "-o",
            "custom",
            "-c",
            "1",
            "--scorers",
            "accuracy",
            "--max-latency",
            "10",
            "--max-cost",
            "1",
        )
    )
    assert result.exit_code == 0
    report = json.loads((workspace / "custom" / "report.json").read_text())
    assert [c["id"] for c in report["cases"]] == ["b"]


def test_run_progress_lines(workspace: Path) -> None:
    result = invoke(
        "run", "--dataset", "golden.jsonl", "--agent", "myagent.py:agent", "--scorers", "accuracy"
    )
    assert "[2/2]" in result.output  # progress goes to stderr (mixed into output by CliRunner)


def test_run_fail_under_and_exit_codes(workspace: Path) -> None:
    ok = invoke(*run_args("--scorers", "accuracy", "--fail-under", "accuracy=0.4"))
    assert ok.exit_code == 0
    low = invoke(
        *run_args(
            "--scorers",
            "accuracy",
            "--fail-under",
            "accuracy=0.9",
            "--fail-under",
            "pass_rate=0.9",
            "--fail-under",
            "nope=1",
        )
    )
    assert low.exit_code == 1
    for fragment in ("accuracy 0.500 < required 0.900", "pass_rate", "no such scorer"):
        assert fragment in low.output
    bad = invoke(*run_args("--fail-under", "garbage"))
    assert bad.exit_code == 2 and "NAME=VALUE" in bad.output


def test_run_fail_under_with_unscored_scorer(workspace: Path) -> None:
    (workspace / "norefs.jsonl").write_text('{"id": "x", "input": "good"}\n')
    result = invoke(
        "run",
        "--dataset",
        "norefs.jsonl",
        "--agent",
        "myagent.py:agent",
        "-q",
        "--scorers",
        "accuracy",
        "--fail-under",
        "accuracy=0.5",
    )
    assert result.exit_code == 1 and "n/a" in result.output


def test_run_usage_errors(workspace: Path) -> None:
    for args, message in [
        (["--agent", "nofile.py:agent"], "not found"),
        (["--scorers", "bogus"], "unknown scorer"),
        (["--judge", "nocolon"], "invalid judge spec"),
        (["--tag", "missing-tag"], "no cases to run"),
        (["--accuracy-mode", "judge"], "requires a judge"),
    ]:
        base = run_args()
        if args[0] == "--agent":
            base[base.index("--agent") + 1] = args[1]
            args = []
        result = invoke(*base, *args)
        assert result.exit_code == 2 and message in result.output, (args, result.output)
    (workspace / "broken.jsonl").write_text("{nope\n")
    result = invoke("run", "--dataset", "broken.jsonl", "--agent", "myagent.py:agent")
    assert result.exit_code == 2 and "invalid JSON" in result.output


def test_run_baseline_workflow(workspace: Path) -> None:
    first = invoke(*run_args("--scorers", "accuracy", "--save-baseline", "base.json"))
    assert first.exit_code == 0 and "baseline written" in first.output
    same = invoke(*run_args("--scorers", "accuracy", "--baseline", "base.json"))
    assert same.exit_code == 0 and "No regressions" in same.output
    # make the agent worse: flip the good case's reference so accuracy drops
    cases = [dict(c) for c in BASELINE_CASES]
    cases[0]["reference"] = "something else entirely"
    (workspace / "golden.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    worse = invoke(
        *run_args("--scorers", "accuracy", "--baseline", "base.json", "--max-regression", "0.1")
    )
    assert worse.exit_code == 1 and "accuracy mean fell" in worse.output
    lenient = invoke(
        *run_args(
            "--scorers",
            "accuracy",
            "--baseline",
            "base.json",
            "--max-regression",
            "0.9",
            "--strict-cases",
        )
    )
    assert lenient.exit_code == 1 and "passed in the baseline but fails now" in lenient.output
    (workspace / "junk.json").write_text("{}")
    broken = invoke(*run_args("--baseline", "junk.json"))
    assert broken.exit_code == 2


def test_run_with_otel_flag(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk")
    result = invoke(*run_args("--scorers", "accuracy", "--otel"))
    assert result.exit_code == 0


def test_report_baseline_compare_commands(workspace: Path) -> None:
    invoke(*run_args("--scorers", "accuracy"))
    report = "evalkit_results/report.json"
    shown = invoke("report", report, "--html", "again.html")
    assert shown.exit_code == 0 and "accuracy" in shown.output
    assert (workspace / "again.html").exists()
    made = invoke("baseline", report, "-o", "b.json")
    assert made.exit_code == 0 and (workspace / "b.json").exists()
    assert "No regressions" in invoke("compare", "b.json", report).output
    data = json.loads((workspace / "b.json").read_text())
    data["scorers"]["accuracy"]["mean"] = 1.0
    (workspace / "high.json").write_text(json.dumps(data))
    failed = invoke("compare", "high.json", report, "--max-regression", "0.1")
    assert failed.exit_code == 1 and "accuracy" in failed.output
    (workspace / "nope.json").write_text("[]")
    assert invoke("report", "nope.json").exit_code == 2
    assert invoke("baseline", "nope.json", "-o", "x.json").exit_code == 2
    assert invoke("compare", "b.json", "nope.json").exit_code == 2


def test_traces_command(workspace: Path) -> None:
    invoke(*run_args("--scorers", "accuracy"))
    listing = invoke("traces")
    assert listing.exit_code == 0 and listing.output.count("steps") == 2
    assert invoke("traces", "--failed").output.strip() == ""


REPLAY_AGENT = """
from evalkit import AgentAdapter
from evalkit.replay import current_plan

class Agent(AgentAdapter):
    name = "replayer"
    def invoke(self, input, recorder):
        plan = current_plan()
        d = plan.next_decision() if plan else None
        if d is not None and d.cached is not None:
            text = d.cached.output
            recorder.record_llm(model="m", prompt="p", output=text, cached=True)
        else:
            sys_prompt = d.override.system if d and d.override else None
            text = "hardened" if sys_prompt else "naive"
            recorder.record_llm(model="m", prompt="p", output=text)
        return text

agent = Agent()
"""


def replay_workspace(workspace: Path) -> str:
    (workspace / "replayer.py").write_text(REPLAY_AGENT)
    invoke(
        "run",
        "--dataset",
        "golden.jsonl",
        "--agent",
        "replayer.py:agent",
        "-q",
        "--scorers",
        "accuracy",
    )
    return TraceStore("evalkit_results/traces.jsonl").read_all()[0].trace_id


def test_replay_list_and_steps(workspace: Path) -> None:
    trace_id = replay_workspace(workspace)
    assert trace_id in invoke("replay", "--list").output
    out = invoke("replay", trace_id[:8], "--list-steps")
    assert out.exit_code == 0 and "llm" in out.output
    assert invoke("replay").exit_code == 2
    missing = invoke("replay", "zzzz")
    assert missing.exit_code == 2 and "no trace matching" in missing.output
    assert invoke("replay", trace_id).exit_code == 2  # needs --agent


def test_replay_override_diff_and_scoring(workspace: Path) -> None:
    trace_id = replay_workspace(workspace)
    result = invoke(
        "replay",
        trace_id,
        "--agent",
        "replayer.py:agent",
        "--step",
        "0",
        "--system",
        "be safe",
        "--dataset",
        "golden.jsonl",
        "--save",
        "re.jsonl",
    )
    assert result.exit_code == 0, result.output
    for fragment in (
        "Final answer CHANGED",
        "naive",
        "hardened",
        "scores (original -> replayed)",
        "new trace id",
    ):
        assert fragment in result.output
    assert len(TraceStore("re.jsonl").read_all()) == 1


def test_replay_plain_and_errors(workspace: Path) -> None:
    trace_id = replay_workspace(workspace)
    plain = invoke("replay", trace_id, "--agent", "replayer.py:agent")
    assert plain.exit_code == 0 and "Final answer unchanged" in plain.output
    (workspace / "sys.txt").write_text("from a file")
    assert (
        invoke(
            "replay",
            trace_id,
            "--agent",
            "replayer.py:agent",
            "--step",
            "0",
            "--system",
            "@sys.txt",
        ).exit_code
        == 0
    )
    bad_step = invoke(
        "replay", trace_id, "--agent", "replayer.py:agent", "--step", "7", "--model", "x"
    )
    assert bad_step.exit_code == 2 and "out of range" in bad_step.output
    unpaired = invoke("replay", trace_id, "--agent", "replayer.py:agent", "--step", "0")
    assert unpaired.exit_code == 2
    assert invoke("replay", trace_id, "--agent", "nofile.py:x").exit_code == 2
    assert (
        invoke(
            "replay",
            trace_id,
            "--agent",
            "replayer.py:agent",
            "--system",
            "@missing.txt",
            "--step",
            "0",
        ).exit_code
        == 2
    )


def test_replay_dataset_without_matching_case(workspace: Path) -> None:
    trace_id = replay_workspace(workspace)
    (workspace / "other.jsonl").write_text('{"id": "elsewhere", "input": "q"}\n')
    result = invoke("replay", trace_id, "--agent", "replayer.py:agent", "--dataset", "other.jsonl")
    assert result.exit_code == 0 and "not in dataset" in result.output


def test_replay_interactive(workspace: Path) -> None:
    trace_id = replay_workspace(workspace)
    answers = "0\n\nbe careful\n\n"  # step, model, system prompt, full prompt
    result = CliRunner().invoke(
        main, ["replay", trace_id, "--agent", "replayer.py:agent", "-i"], input=answers
    )
    assert result.exit_code == 0, result.output
    assert "hardened" in result.output
    skipped = CliRunner().invoke(
        main, ["replay", trace_id, "--agent", "replayer.py:agent", "-i"], input="\n"
    )
    assert skipped.exit_code == 0 and "Final answer unchanged" in skipped.output
