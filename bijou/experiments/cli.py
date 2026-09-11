"""The bijou command. Every experiment enters here.

Commands that do not need the model stack import it lazily, so config, skill
inspection and run records work in an environment without torch.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from bijou import __version__
from bijou.core.config import load
from bijou.core.runs import RunRecord, load_records
from bijou.core.types import BijouError, SkillReport
from bijou.skills import KNOWN

console = Console()
err = Console(stderr=True)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "Modular capability deltas for masked diffusion language models.\n\n"
        "Start with [bold]bijou skill list[/bold] and [bold]bijou config[/bold]; "
        "neither needs a GPU."
    ),
)
skill_app = typer.Typer(no_args_is_help=True, help="Inspect and train skills.")
run_app = typer.Typer(no_args_is_help=True, help="Inspect run records.")
app.add_typer(skill_app, name="skill")
app.add_typer(run_app, name="run")


def _fail(exc: BijouError) -> None:
    """Report a known failure without a traceback."""
    err.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


def _version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version.")
    ] = False,
) -> None:
    """Root options."""


# ---------- configuration ----------


def _flatten(section: str, values: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for key, value in values.items():
        shown = ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
        rows.append((section, key, shown))
    return rows


@app.command()
def config(
    section: Annotated[str | None, typer.Argument(help="Show only this section.")] = None,
) -> None:
    """Show the resolved configuration, after bijou.toml and BIJOU_* env."""
    try:
        cfg = load()
    except BijouError as exc:
        _fail(exc)
    table = Table(title="resolved configuration", title_justify="left")
    table.add_column("section", style="dim")
    table.add_column("key")
    table.add_column("value", overflow="fold")
    dumped = cfg.model_dump(mode="json")
    if section and section not in dumped:
        err.print(f"[red]error[/red] unknown section {section}; try: {', '.join(dumped)}")
        raise typer.Exit(1)
    for name, values in dumped.items():
        if section and name != section:
            continue
        for row in _flatten(name, values):
            table.add_row(*row)
    console.print(table)


# ---------- skills ----------


@skill_app.command("list")
def skill_list() -> None:
    """Every known skill and the split sizes it will be run with."""
    cfg = load()
    table = Table(title="skills", title_justify="left")
    table.add_column("skill")
    table.add_column("train", justify="right")
    table.add_column("eval", justify="right")
    table.add_column("adapter")
    for name in KNOWN:
        path = cfg.paths.adapters / f"{name}.pt"
        trained = "[green]trained[/green]" if path.exists() else "[dim]not trained[/dim]"
        table.add_row(name, f"{cfg.train.train_samples:,}", f"{cfg.eval.eval_samples:,}", trained)
    console.print(table)


@skill_app.command("names")
def skill_names() -> None:
    """One skill name per line, for scripting."""
    for name in KNOWN:
        typer.echo(name)


@skill_app.command("sample")
def skill_sample(
    name: Annotated[str, typer.Argument(help="Skill to sample from.")],
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="How many samples.")] = 3,
    seed: Annotated[int | None, typer.Option(help="Override the eval seed.")] = None,
) -> None:
    """Show generated samples for one skill. Loads no model."""
    from bijou.skills import load as load_skill

    try:
        skill = load_skill(name)
    except BijouError as exc:
        _fail(exc)
    for sample in skill.generate(n, load().eval.seed if seed is None else seed):
        console.print(
            Panel(
                f"{sample.prompt}\n\n[bold green]{sample.target}[/bold green]",
                title=sample.id,
                title_align="left",
                border_style="dim",
            )
        )


@skill_app.command("grade")
def skill_grade(
    name: Annotated[str, typer.Argument(help="Skill whose grader to run.")],
    output: Annotated[str, typer.Argument(help="A candidate model output.")],
    index: Annotated[int, typer.Option(help="Which eval sample to grade against.")] = 0,
) -> None:
    """Grade one candidate output against one sample. Loads no model."""
    from bijou.skills import load as load_skill

    try:
        skill = load_skill(name)
    except BijouError as exc:
        _fail(exc)
    sample = skill.generate(index + 1, load().eval.seed)[index]
    score = skill.grade(sample, output)
    mark = "[green]pass[/green]" if score.passed else "[red]fail[/red]"
    console.print(f"{mark}  value={score.value:.2f}  {score.detail}")


@skill_app.command("train")
def skill_train(
    name: Annotated[str, typer.Argument(help="Skill to train an adapter for.")],
    full_finetune: Annotated[
        bool, typer.Option("--full-finetune", help="Train the base weights instead.")
    ] = False,
) -> None:
    """Train one adapter, or the full-finetune upper bound. Needs the model stack."""
    from bijou.runtime.train import train_adapter

    try:
        path = train_adapter(load(), name, full_finetune=full_finetune)
    except BijouError as exc:
        _fail(exc)
    console.print(f"wrote [bold]{path}[/bold]")


# ---------- experiments ----------


@app.command()
def evaluate(
    skills: Annotated[
        list[str] | None, typer.Option("--skill", help="Restrict to these skills.")
    ] = None,
) -> None:
    """Score the composition matrix. Needs the model stack and trained adapters."""
    from bijou.experiments import matrix

    cfg = load()
    if skills:
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"skills": tuple(skills)})})
    try:
        _matrix(matrix.run(cfg))
    except BijouError as exc:
        _fail(exc)


def _matrix(reports: list[SkillReport]) -> None:
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


# ---------- runs ----------


@run_app.command("list")
def run_list(
    kind: Annotated[str | None, typer.Option(help="Filter to train or eval.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Show the newest N.")] = 20,
) -> None:
    """Every run record, newest first."""
    cfg = load()
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
    for key in ("final_loss", "accuracy"):
        if key in record.scores:
            return f"{key}={record.scores[key]:.4f}"
    if len(record.scores) == 1:
        key, value = next(iter(record.scores.items()))
        return f"{key}={value:g}"
    return f"{len(record.scores)} scores"


@run_app.command("show")
def run_show(run_id: Annotated[str, typer.Argument(help="A run id from bijou run list.")]) -> None:
    """The full record for one run."""
    cfg = load()
    path = cfg.paths.runs / run_id / "record.json"
    if not path.exists():
        err.print(f"[red]error[/red] no run {run_id} in {cfg.paths.runs}")
        raise typer.Exit(1)
    console.print(Syntax(path.read_text(), "json", theme="ansi_dark", word_wrap=True))


if __name__ == "__main__":
    app()
