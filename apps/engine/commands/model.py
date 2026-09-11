"""Commands for the diffusion model: config, skills, the bank, collection, the matrix, runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from engine.commands.app import app
from engine.core.config import load
from engine.core.runs import RunRecord, load_records
from engine.core.types.errors import EngineError
from engine.core.types.model import SkillReport
from engine.skills import names as skill_names

console = Console()
err = Console(stderr=True)

skill_app = typer.Typer(no_args_is_help=True, help="Inspect and train skills.")
run_app = typer.Typer(no_args_is_help=True, help="Inspect run records.")
collect_app = typer.Typer(no_args_is_help=True, help="Collect data for new skills.")
app.add_typer(skill_app, name="skill")
app.add_typer(run_app, name="runs")
app.add_typer(collect_app, name="collect")


def _fail(exc: EngineError) -> None:
    """Report a known failure without a traceback."""
    err.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


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
    except EngineError as exc:
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
    """Every known skill, built in or collected, and what is trained for it."""
    cfg = load()
    table = Table(title="skills", title_justify="left")
    table.add_column("skill")
    table.add_column("kind")
    table.add_column("adapter")
    table.add_column("full fine-tune")

    def mark(present: bool) -> str:
        return "[green]trained[/green]" if present else "[dim]not trained[/dim]"

    from engine.skills import KNOWN

    for name in skill_names(cfg.paths.data):
        table.add_row(
            name,
            "built in" if name in KNOWN else "collected",
            mark(cfg.adapter_path(name).exists()),
            mark(cfg.full_finetune_path(name).exists()),
        )
    console.print(table)


@skill_app.command("names")
def skill_names_command() -> None:
    """One skill name per line, for scripting."""
    for name in skill_names(load().paths.data):
        typer.echo(name)


@skill_app.command("sample")
def skill_sample(
    name: Annotated[str, typer.Argument(help="Skill to sample from.")],
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="How many samples.")] = 3,
    seed: Annotated[int | None, typer.Option(help="Override the eval seed.")] = None,
) -> None:
    """Show eval samples for one skill. Loads no model."""
    from engine.skills import load as load_skill

    cfg = load()
    try:
        skill = load_skill(name, cfg.paths.data)
        samples = skill.generate(n, cfg.eval.seed if seed is None else seed, split="eval")
    except EngineError as exc:
        _fail(exc)
    for sample in samples:
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
    """Grade one candidate output against one eval sample. Loads no model."""
    from engine.skills import load as load_skill

    cfg = load()
    try:
        skill = load_skill(name, cfg.paths.data)
        samples = skill.generate(index + 1, cfg.eval.seed, split="eval")
    except EngineError as exc:
        _fail(exc)
    if index >= len(samples):
        err.print(f"[red]error[/red] {name} has {len(samples)} eval samples")
        raise typer.Exit(1)
    score = skill.grade(samples[index], output)
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
    from engine.runtime.train import train_adapter

    try:
        path = train_adapter(load(), name, full_finetune=full_finetune)
    except EngineError as exc:
        _fail(exc)
    console.print(f"wrote [bold]{path}[/bold]")


# ---------- the skill server ----------


@app.command("serve-skills")
def serve_skills() -> None:
    """Serve the skill bank over HTTP, for an agent elsewhere. Needs the model stack."""
    from engine.routes.skills import serve as run_server

    try:
        run_server(load())
    except EngineError as exc:
        _fail(exc)


# ---------- collection ----------


@collect_app.command("list")
def collect_list() -> None:
    """Every skill spec waiting under the proposals directory, and whether it is approved."""
    from engine.collect.spec import list_specs

    cfg = load()
    specs = list_specs(cfg.proposals_dir)
    if not specs:
        console.print(f"[dim]no specs in {cfg.proposals_dir}[/dim]")
        return
    table = Table(title=f"skill specs in {cfg.proposals_dir}", title_justify="left")
    for column in ("file", "skill", "approved", "source", "seen", "description"):
        table.add_column(column, overflow="fold")
    for path, spec in specs:
        if spec is None:
            table.add_row(path.name, "[red]invalid[/red]", "", "", "", "")
            continue
        table.add_row(
            path.name,
            spec.name,
            "[green]yes[/green]" if spec.approved else "[yellow]no[/yellow]",
            spec.source,
            str(spec.occurrences),
            spec.description,
        )
    console.print(table)


@collect_app.command("new")
def collect_new(
    name: Annotated[str, typer.Argument(help="Lower snake case skill name.")],
    description: Annotated[str, typer.Option(help="What the skill does.")],
    instruction: Annotated[str, typer.Option(help="The instruction every input is given with.")],
) -> None:
    """Write an unapproved spec by hand. Edit it, add pairs, then set approved to true."""
    from engine.collect.spec import SkillSpec, SpecError, write_spec

    cfg = load()
    path = cfg.proposals_dir / f"{name}.json"
    try:
        write_spec(SkillSpec(name=name, description=description, instruction=instruction), path)
    except (SpecError, ValueError) as exc:
        err.print(f"[red]error[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"wrote [bold]{path}[/bold]; review it and set approved to true to collect")


@collect_app.command("run")
def collect_run(
    spec_path: Annotated[Path, typer.Argument(help="A spec file, usually under data/proposals.")],
    n: Annotated[int | None, typer.Option("--n", "-n", min=3, help="Examples to collect.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing dataset skill.")] = False,
) -> None:
    """Collect an approved spec into a dataset skill with the teacher model."""
    from engine.collect.pipeline import collect
    from engine.collect.spec import read_spec
    from engine.collect.teacher import OpenAITeacher

    cfg = load()
    try:
        spec = read_spec(spec_path)
        out = collect(cfg, spec, OpenAITeacher(cfg.collect), n=n, overwrite=overwrite)
    except EngineError as exc:
        _fail(exc)
    console.print(f"wrote [bold]{out}[/bold]; next: just skill train {spec.name}")


# ---------- experiments ----------


@app.command()
def evaluate(
    skills: Annotated[
        list[str] | None, typer.Option("--skill", help="Restrict to these skills.")
    ] = None,
) -> None:
    """Score the composition matrix. Needs the model stack and trained adapters."""
    from engine.experiments import matrix

    cfg = load()
    if skills:
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"skills": tuple(skills)})})
    try:
        _matrix(matrix.run(cfg))
    except EngineError as exc:
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
    kind: Annotated[str | None, typer.Option(help="Filter to train, eval or collect.")] = None,
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
    for key in ("final_loss", "accuracy", "collected"):
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
