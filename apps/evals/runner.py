"""Loads the golden cases and runs each one against a serving engine over HTTP.

A case that waits on a confirmation is scored as it stands; evals never approves an action.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from pydantic import ValidationError

from evals.core.types import EvalCase, EvalsError, RunView


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Every case in every jsonl file under cases_dir, in file then line order."""
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate_json(line))
            except ValidationError as exc:
                raise EvalsError(f"{path}:{number} is not a case: {exc}") from exc
    if not cases:
        raise EvalsError(f"no cases under {cases_dir}")
    ids = [c.id for c in cases]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise EvalsError(f"case ids used twice: {', '.join(duplicated)}")
    return cases


def run_case(case: EvalCase, client: httpx.Client) -> RunView:
    """One request through the engine."""
    try:
        response = client.post("/run", json={"request": case.request, "user_id": f"eval-{case.id}"})
    except httpx.HTTPError as exc:
        raise EvalsError(
            f"engine at {client.base_url} unreachable: {exc}; start it: just serve"
        ) from exc
    if response.status_code != 200:
        raise EvalsError(
            f"case {case.id}: engine returned {response.status_code}: {response.text[:200]}"
        )
    return RunView.model_validate(response.json())
