"""evals run, evals compare, evals baseline, evals cases."""

from __future__ import annotations

import json
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from evals import report as reports
from evals import suites
from evals.core.config import load
from evals.core.types import CaseResult, EvalReport, EvalsError
from evals.runner import load_cases, run_case

console = Console()
err = Console(stderr=True)
app = typer.Typer(no_args_is_help=True, add_completion=False, help="Harness evals.")


def _fail(exc: EvalsError) -> None:
    err.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


def _print(report: EvalReport) -> None:
    table = Table(
        title=f"evals against {report.engine_url} at {report.git_sha}", title_justify="left"
    )
    for column in ("suite", "passed", "rate"):
        table.add_column(column, justify="right" if column != "suite" else "left")
    for s in report.suites:
        table.add_row(s.suite, f"{s.passed}/{s.total}", f"{s.rate:.0%}")
    console.print(table)
    for r in report.results:
        if not r.score.passed:
            console.print(
                f"  [red]fail[/red] {r.suite} {r.case_id} ({r.session_id}): {r.score.detail}"
            )


@app.command()
def cases() -> None:
    """Every golden case and the suites it is scored on."""
    cfg = load()
    try:
        listed = load_cases(cfg.cases_dir)
    except EvalsError as exc:
        _fail(exc)
    table = Table(title=f"cases in {cfg.cases_dir}", title_justify="left")
    for column in ("case", "suites", "request"):
        table.add_column(column, overflow="fold")
    for case in listed:
        table.add_row(case.id, ", ".join(case.suites), case.request)
    console.print(table)


@app.command()
def run(
    suite: Annotated[list[str] | None, typer.Option("--suite", help="Only these suites.")] = None,
    case: Annotated[list[str] | None, typer.Option("--case", help="Only these case ids.")] = None,
) -> None:
    """Run the golden cases against the live engine, score them, and write the report."""
    cfg = load()
    wanted = set(suite or suites.NAMES)
    try:
        for name in wanted:
            suites.load(name)
        selected = [
            c
            for c in load_cases(cfg.cases_dir)
            if wanted & set(c.suites) and (not case or c.id in case)
        ]
        results: list[CaseResult] = []
        with httpx.Client(base_url=cfg.engine_url, timeout=cfg.timeout_secs) as client:
            for c in selected:
                console.print(f"[dim]{c.id}[/dim] {c.request}")
                results += reports.score(c, run_case(c, client), wanted)
    except EvalsError as exc:
        _fail(exc)
    report = reports.build(results, cfg.engine_url, len(selected))
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(report.model_dump_json(indent=2))
    _print(report)
    console.print(f"report: {cfg.report_path}")


@app.command()
def baseline() -> None:
    """Promote the last report's suite pass counts to the committed baseline."""
    cfg = load()
    try:
        report = reports.read_report(cfg.report_path)
    except EvalsError as exc:
        _fail(exc)
    cfg.baseline_path.write_text(json.dumps(reports.baseline_of(report), indent=2) + "\n")
    console.print(f"baseline: {cfg.baseline_path}")


@app.command()
def compare(
    tolerance: Annotated[float | None, typer.Option(help="Allowed drop in pass rate.")] = None,
) -> None:
    """Fail when any suite's pass rate fell below the baseline."""
    cfg = load()
    try:
        report = reports.read_report(cfg.report_path)
        regressions = reports.compare(
            report,
            reports.read_baseline(cfg.baseline_path),
            cfg.tolerance if tolerance is None else tolerance,
        )
    except EvalsError as exc:
        _fail(exc)
    if regressions:
        err.print("[red]regressions[/red] " + "; ".join(regressions))
        raise typer.Exit(1)
    console.print("[green]no regressions against the baseline[/green]")


if __name__ == "__main__":
    app()
