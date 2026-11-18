# Contributing to EvalKit

Thanks for helping! This guide covers setup, style, testing and the PR checklist. By taking part
you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

```bash
git clone https://github.com/depuc8/EvalKit && cd EvalKit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest                      # runs the suite and enforces >= 90% coverage
```

Python 3.10+ is supported; CI tests 3.10 through 3.14.

## Where to start

- Bug fix or small improvement: open a PR directly.
- New scorer, adapter, CLI command or behaviour change: **open an issue first** so we can agree on
  the shape before you spend time on it.
- New framework adapter (ADK, CrewAI, ...): see [docs/adapters.md](docs/adapters.md). Adapters that
  need a heavy dependency go behind an optional extra and are imported lazily.

## Code style

- Formatting and linting: **ruff** (`ruff format` and `ruff check`); line length 100.
- Types: **mypy --strict** must pass on `src/evalkit`. Annotate everything; avoid `Any` unless the
  value is genuinely dynamic (LLM output, framework objects). Justify any `# type: ignore` with the
  specific error code.
- Match the surrounding code: naming, docstring style (short, imperative, explain *why* and limits),
  comment density. Don't add comments that restate the code.
- Keep the core dependency list tiny (`pydantic`, `click`). Framework and provider integrations are
  optional extras.
- Public API goes through `evalkit/__init__.py` (lazy exports) - update `_EXPORTS` and the
  `TYPE_CHECKING` imports together.

```bash
ruff format src tests examples && ruff check src tests examples && mypy
```

## Testing rules

- Every behaviour change comes with tests; every bug fix comes with a test that fails without it.
- Coverage must stay at or above **90%** (CI enforces it).
- Tests must be **deterministic and offline**: no network, no API keys, no sleeping for
  timing-dependent assertions. Use fakes for judges (`lambda prompt: '{"score": 1}'`) and the
  offline example model for LangGraph tests.
- Scorers: test the passing case, the failing case, the skipped case, and the judge-failure case
  (a failed judge must produce `error` and a failed score, never a silent pass).
- Don't weaken a threshold or an assertion to make a test pass; fix the cause or discuss it.
- The example (`examples/rag_agent`) is tested too; if you change the agent, scorers or dataset,
  regenerate the baseline deliberately (`evalkit run ... --save-baseline examples/rag_agent/baseline.json`)
  and explain the change in the PR.

## Docs

Docs live in `docs/` (mkdocs). Behaviour changes need a docs update; each scorer page must state
its limits honestly. Preview with `mkdocs serve`; CI builds with `mkdocs build --strict`.

## Commits and PRs

- Small, focused PRs; one logical change each.
- Commit messages: imperative mood, a short subject line, a body explaining *why* if not obvious.
- Add an entry under **Unreleased** in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- We use [Semantic Versioning](https://semver.org/): breaking changes to the public API, CLI flags,
  report/trace/baseline JSON formats bump the major version (minor while `0.x`).

### PR checklist

- [ ] Tests added/updated; `pytest` passes locally with coverage >= 90%
- [ ] `ruff format`, `ruff check` and `mypy` are clean
- [ ] Docs updated (and `mkdocs build --strict` passes) for user-facing changes
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] No secrets, API keys or private data in code, fixtures or traces
- [ ] Format changes (trace / report / baseline JSON) are backwards compatible, or called out

## Releasing (maintainers)

1. Move *Unreleased* entries under a new version heading in `CHANGELOG.md`; bump
   `src/evalkit/_version.py`.
2. Merge, then tag: `git tag vX.Y.Z && git push --tags`.
3. The `release` workflow verifies the tag matches the version, builds, and publishes to PyPI via
   trusted publishing.

## Security

Please report vulnerabilities privately through the repository's **Security** tab (GitHub private
vulnerability reporting), not in public issues.
