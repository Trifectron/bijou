"""Which diffusion skills the selector equipped."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def score(case: EvalCase, run: RunView) -> Score | None:
    equipped = sorted(run.skills)
    if case.expect.no_skills:
        return Score(passed=not equipped, detail=f"expected none, equipped {equipped}")
    want = case.expect.skills
    if want is None:
        return None
    missing = [s for s in want if s not in equipped]
    return Score(passed=not missing, detail=f"expected {want}, equipped {equipped}")
