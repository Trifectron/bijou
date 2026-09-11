"""How the run ended."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def score(case: EvalCase, run: RunView) -> Score | None:
    want = case.expect.status
    if want is None:
        return None
    return Score(passed=run.status == want, detail=f"expected {want}, got {run.status}")
