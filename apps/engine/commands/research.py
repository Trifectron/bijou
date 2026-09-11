"""engine matrix, engine runs, engine config: the research eval and what it recorded."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.syntax import Syntax
from rich.table import Table

from engine.commands.app import app, console, fail, settings
from engine.core.runs import RunRecord, load_records
from engine.core.types.diffusion import SkillReport
from engine.core.types.errors import EngineError


@app.command()
def matrix(
    train: Annotated[
        bool, typer.Option("--train", help="Train every adapter and full fine-tune first.")
    ] = False,
    skills: Annotated[
        list[str] | None, typer.Option("--skill", help="Restrict to these skills.")
    ] = None,
) -> None:
    """Score the composition matrix. Needs the model stack and trained adapters."""
    from engine.runtime import matrix as composition

    cfg = settings()
    if skills:
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"skills": tuple(skills)})})
    try:
        if train:
            from engine.runtime.train import train_adapter

            for skill in cfg.eval.skills:
                console.print(f"training {skill}")
                train_adapter(cfg, skill)
                train_adapter(cfg, skill, full_finetune=True)
        _table(composition.run(cfg))
    except EngineError as exc:
        fail(exc)


def _table(reports: list[SkillReport]) -> None:
    """Conditions as rows, skill evals as columns. Off-diagonal cells are the damage."""
    columns = sorted({r.skill for r in reports})
    by_condition: dict[str, dict[str, SkillReport]] = {}
    for report in reports:
        by_condition.setdefault(report.condition, {})[report.skill] = report
    table = Table(title="composition matrix (pass rate)", title_justify="left")
    table.add_column("condition")
    for skill in columns:
        table.add_column(skill, justify="right")
    for condition, row in by_condition.items():
        cells = []
        for skill in columns:
            if skill not in row:
                cells.append("[dim]-[/dim]")
                continue
            rate = f"{row[skill].rate:.0%}"
            cells.append(f"[bold]{rate}[/bold]" if skill in condition else rate)
        table.add_row(condition, *cells)
    console.print(table)
    console.print(
        "[dim]bold is a skill's own adapter; the other cells in that row are its damage[/dim]"
    )


@app.command()
def runs(
    run_id: Annotated[str | None, typer.Argument(help="Print one run record in full.")] = None,
    kind: Annotated[str | None, typer.Option(help="Only train, eval or collect.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="The newest N.")] = 20,
) -> None:
    """Every run record, newest first, or one in full."""
    cfg = settings()
    if run_id is not None:
        path = cfg.paths.runs / run_id / "record.json"
        if not path.exists():
            fail(FileNotFoundError(f"no run {run_id} in {cfg.paths.runs}"))
        console.print(Syntax(path.read_text(), "json", theme="ansi_dark", word_wrap=True))
        return
    records = [r for r in reversed(load_records(cfg.paths.runs)) if kind in (None, r.kind)]
    if not records:
        console.print(f"[dim]no runs in {cfg.paths.runs}[/dim]")
        return
    table = Table(title=f"runs in {cfg.paths.runs}", title_justify="left")
    for column in ("run", "kind", "git", "started", "notes", "headline"):
        table.add_column(column, overflow="fold")
    for record in records[:limit]:
        table.add_row(
            record.run_id,
            record.kind,
            record.env.get("git_sha", "?"),
            record.started_at[:19],
            record.notes,
            _headline(record),
        )
    console.print(table)


def _headline(record: RunRecord) -> str:
    """The one score worth putting in a list, or how many there are."""
    if not record.scores:
        return "[dim]none[/dim]"
    for key in ("final_loss", "accuracy", "collected"):
        if key in record.scores:
            return f"{key}={record.scores[key]:.4f}"
    if len(record.scores) == 1:
        key, value = next(iter(record.scores.items()))
        return f"{key}={value:g}"
    return f"{len(record.scores)} scores"


def _flatten(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows = []
        for key, inner in value.items():
            rows += _flatten(f"{prefix}.{key}" if prefix else key, inner)
        return rows
    shown = ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    return [(prefix, shown)]


@app.command()
def config(
    table: Annotated[str | None, typer.Argument(help="Only this table, as in agent.llm.")] = None,
) -> None:
    """The resolved configuration, after bijou.toml and BIJOU_* env. Secrets are hidden."""
    rows = _flatten("", settings().model_dump(mode="json"))
    if table:
        rows = [(k, v) for k, v in rows if k == table or k.startswith(table + ".")]
        if not rows:
            fail(KeyError(f"no table {table}"))
    out = Table(title="resolved configuration", title_justify="left")
    out.add_column("key")
    out.add_column("value", overflow="fold")
    for key, value in rows:
        out.add_row(key, value)
    console.print(out)
