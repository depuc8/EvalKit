# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- LangGraph adapter on Python 3.10: async runs dropped tool, LLM and retriever steps from traces
  (LangChain loses its callback context before Python 3.11). The async entry point now uses the
  synchronous path on 3.10.

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

[Unreleased]: https://github.com/depuc8/EvalKit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/depuc8/EvalKit/releases/tag/v0.1.0
