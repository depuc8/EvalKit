# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-18

Initial release.

### Added
- Trace capture: framework-neutral `TraceRecorder`, `@traced` / `trace_run`, JSONL `TraceStore`,
  and an optional OpenTelemetry span exporter.
- LangGraph adapter: callback capturing nodes, LLM calls (prompts, outputs, tokens, time to first
  token), tool calls and retrievals; `LangGraphAdapter` for running compiled graphs.
- Scorers: accuracy (exact, fuzzy, LLM judge), groundedness, safety, latency & cost, tool use;
  `Scorer` base class for custom scorers; Anthropic LLM judge.
- Datasets from JSONL and CSV.
- `evalkit run`: concurrent runs with timeouts, terminal / JSON / static HTML reports.
- `evalkit replay`: step-by-step replay with prompt/model overrides at an LLM step, trace diff,
  before/after scoring.
- CI: `assert_eval` and pytest plugin, baselines with regression detection, `--fail-under`,
  example GitHub Actions workflow.
- Example LangGraph RAG agent with a 50-case golden dataset (runs offline).

[Unreleased]: https://github.com/evalkit/evalkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/evalkit/evalkit/releases/tag/v0.1.0
