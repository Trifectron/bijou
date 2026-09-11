"""How long the run took, as the engine measured it."""

from __future__ import annotations

from evals.core.types import EvalCase, RunView, Score


def score(case: EvalCase, run: RunView) -> Score | None:
    limit = case.expect.max_latency_ms
    if limit is None:
        return None
    return Score(passed=run.duration_ms <= limit, detail=f"{run.duration_ms}ms, limit {limit}ms")
