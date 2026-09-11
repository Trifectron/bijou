"""Which tools the subagents called."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def matches(pattern: str, name: str) -> bool:
    """Exact, or a prefix when the pattern ends in *."""
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def score(case: EvalCase, run: RunView) -> Score | None:
    called = sorted(run.tools)
    if case.expect.no_tools:
        return Score(passed=not called, detail=f"expected none, called {called}")
    want = case.expect.tools
    if want is None:
        return None
    hit = any(matches(p, name) for p in want for name in called)
    return Score(passed=hit, detail=f"expected one of {want}, called {called}")
