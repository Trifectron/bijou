"""Scoring runs into a report, and comparing a report against the committed baseline.

The baseline is {suite: {passed, total}}. A suite in the baseline and missing from the report is
a regression; a suite new to the report is shown and not compared.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evals import suites
from evals.core.types import CaseResult, EvalCase, EvalReport, EvalsError, RunView, SuiteReport


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def score(case: EvalCase, run: RunView, wanted: set[str]) -> list[CaseResult]:
    """Every suite the case names, and the run asks for, that has something to check."""
    results = []
    for name in case.suites:
        if name not in wanted:
            continue
        found = suites.load(name).score(case, run)
        if found is not None:
            results.append(
                CaseResult(case_id=case.id, suite=name, score=found, session_id=run.session_id)
            )
    return results


def build(results: list[CaseResult], engine_command: list[str], cases: int) -> EvalReport:
    reports = []
    for name in suites.NAMES:
        mine = [r for r in results if r.suite == name]
        if mine:
            reports.append(
                SuiteReport(suite=name, passed=sum(r.score.passed for r in mine), total=len(mine))
            )
    return EvalReport(
        engine_command=engine_command,
        git_sha=git_sha(),
        started_at=datetime.now(UTC).isoformat(),
        cases=cases,
        results=results,
        suites=reports,
    )


def baseline_of(report: EvalReport) -> dict[str, dict[str, int]]:
    return {s.suite: {"passed": s.passed, "total": s.total} for s in report.suites}


def compare(report: EvalReport, baseline: dict[str, dict[str, int]], tolerance: float) -> list[str]:
    """Every suite whose pass rate fell more than tolerance below the baseline."""
    now = {s.suite: s for s in report.suites}
    regressions = []
    for name in sorted(set(now) | set(baseline)):
        before, current = baseline.get(name), now.get(name)
        if before is None or not before.get("total"):
            continue
        if current is None:
            regressions.append(f"{name}: in the baseline, missing from this report")
            continue
        rate = before["passed"] / before["total"]
        if current.rate + tolerance < rate:
            regressions.append(f"{name}: {rate:.0%} -> {current.rate:.0%}")
    return regressions


def read_report(path: Path) -> EvalReport:
    if not path.exists():
        raise EvalsError(f"no report at {path}; run: just evals run")
    return EvalReport.model_validate_json(path.read_text())


def read_baseline(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        raise EvalsError(f"no baseline at {path}; run evals run, then evals baseline")
    data: dict[str, dict[str, int]] = json.loads(path.read_text())
    return data
