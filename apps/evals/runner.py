"""Loads the golden cases and runs each one through engine run --json in a subprocess.

A case that waits on a confirmation is scored as it stands; evals never approves an action.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

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


def run_case(case: EvalCase, engine_command: list[str], timeout_secs: float) -> RunView:
    """One request through the engine command, as user eval-<case id>."""
    argv = [*engine_command, "run", "--json", "--user", f"eval-{case.id}", case.request]
    shown = shlex.join(engine_command)
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_secs, check=False
        )
    except FileNotFoundError as exc:
        raise EvalsError(
            f"engine command {shown} not found: {exc}; set evals.engine_command or run: just setup"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EvalsError(
            f"case {case.id}: engine gave no result in {timeout_secs:g}s; "
            "check llama-server is up (just up model) and the request runs with: just agent"
        ) from exc
    if done.returncode != 0:
        raise EvalsError(
            f"case {case.id}: {shown} exited {done.returncode}: {done.stderr.strip()[-500:]}; "
            "check llama-server is up (just up model) and the request runs with: just agent"
        )
    try:
        return RunView.model_validate_json(done.stdout)
    except ValidationError as exc:
        raise EvalsError(
            f"case {case.id}: {shown} printed no RunResult: {done.stdout.strip()[:200]!r}"
        ) from exc
