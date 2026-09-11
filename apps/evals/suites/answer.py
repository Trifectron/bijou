"""What the answer says."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def score(case: EvalCase, run: RunView) -> Score | None:
    want = case.expect.answer_contains
    if not want:
        return None
    text = run.answer.lower()
    missing = [w for w in want if w.lower() not in text]
    return Score(passed=not missing, detail=f"missing {missing}" if missing else "")
