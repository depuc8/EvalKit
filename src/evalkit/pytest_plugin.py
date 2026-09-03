"""Pytest plugin (auto-registered via the ``pytest11`` entry point).

Provides the ``assert_eval`` fixture::

    def test_agent_quality(assert_eval):
        results = run_eval(cases, adapter, scorers)
        assert_eval(results, min_accuracy=0.85)

Everything is imported lazily so having EvalKit installed adds no cost to other test runs.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture(name="assert_eval")
def evalkit_assert_eval() -> Callable[..., None]:
    """The :func:`evalkit.assertions.assert_eval` function as a fixture."""
    from evalkit.assertions import assert_eval

    return assert_eval
