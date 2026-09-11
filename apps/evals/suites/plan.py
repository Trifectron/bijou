"""How many steps the planner made, and that planning did not fall back."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def score(case: EvalCase, run: RunView) -> Score | None:
    low, high = case.expect.min_steps, case.expect.max_steps
    if low is None and high is None:
        return None
    n = len(run.plan.steps)
    fits = (low is None or n >= low) and (high is None or n <= high)
    passed = fits and not run.plan.fallback
    detail = f"{n} steps, wanted {low or 0} to {high or 'any'}"
    if run.plan.fallback:
        detail += "; planning fell back"
    return Score(passed=passed, detail=detail)
