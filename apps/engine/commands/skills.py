"""engine skills: everything about the skill bank, from a proposal to a trained adapter.

list      every skill, built in or collected, and what is trained for it
sample    eval samples of one skill
grade     one output against one eval sample
train     one adapter, or the full fine-tune upper bound
propose   recurring work no skill covered, from the sessions; --write saves each as a spec
specs     the specs waiting under the proposals directory
new       a spec written by hand
collect   an approved spec into a dataset skill, with the teacher model
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from engine.collect.spec import SkillSpec, list_specs, read_spec, write_spec
from engine.commands.app import app, console, fail, settings
from engine.core.types.errors import EngineError
from engine.memory.patterns import PatternMiner, write_proposals
from engine.memory.sessions import SqliteSessionStore
from engine.skills import KNOWN
from engine.skills import load as load_skill
from engine.skills import names as skill_names

skills_app = typer.Typer(no_args_is_help=True, help="The skill bank: list, train, collect.")
app.add_typer(skills_app, name="skills")


def _mark(present: bool) -> str:
    return "[green]trained[/green]" if present else "[dim]not trained[/dim]"


@skills_app.command("list")
def list_skills() -> None:
    """Every skill, built in or collected, and what is trained for it."""
    cfg = settings()
    table = Table(title="skills", title_justify="left")
    for column in ("skill", "kind", "adapter", "full fine-tune", "description"):
        table.add_column(column, overflow="fold")
    for name in skill_names(cfg.paths.data):
        table.add_row(
            name,
            "built in" if name in KNOWN else "collected",
            _mark(cfg.adapter_path(name).exists()),
            _mark(cfg.full_finetune_path(name).exists()),
            load_skill(name, cfg.paths.data).DESCRIPTION,
        )
    console.print(table)


@skills_app.command("names", hidden=True)
def list_names() -> None:
    """One skill name per line, for scripting."""
    for name in skill_names(settings().paths.data):
        typer.echo(name)


@skills_app.command()
def sample(
    name: Annotated[str, typer.Argument(help="Skill to sample from.")],
    n: Annotated[int, typer.Option("--n", "-n", min=1, help="How many samples.")] = 3,
    seed: Annotated[int | None, typer.Option(help="Override the eval seed.")] = None,
) -> None:
    """Show eval samples of one skill. Loads no model."""
    cfg = settings()
    try:
        skill = load_skill(name, cfg.paths.data)
        samples = skill.generate(n, cfg.eval.seed if seed is None else seed, split="eval")
    except EngineError as exc:
        fail(exc)
    for s in samples:
        console.print(
            Panel(
                f"{s.prompt}\n\n[bold green]{s.target}[/bold green]",
                title=s.id,
                title_align="left",
                border_style="dim",
            )
        )


@skills_app.command()
def grade(
    name: Annotated[str, typer.Argument(help="Skill whose grader to run.")],
    output: Annotated[str, typer.Argument(help="A candidate model output.")],
    index: Annotated[int, typer.Option(help="Which eval sample to grade against.")] = 0,
) -> None:
    """Grade one output against one eval sample. Loads no model."""
    cfg = settings()
    try:
        skill = load_skill(name, cfg.paths.data)
        samples = skill.generate(index + 1, cfg.eval.seed, split="eval")
    except EngineError as exc:
        fail(exc)
    if index >= len(samples):
        fail(ValueError(f"{name} has {len(samples)} eval samples"))
    score = skill.grade(samples[index], output)
    mark = "[green]pass[/green]" if score.passed else "[red]fail[/red]"
    console.print(f"{mark}  value={score.value:.2f}  {score.detail}")


@skills_app.command()
def train(
    name: Annotated[str, typer.Argument(help="Skill to train an adapter for.")],
    full: Annotated[bool, typer.Option("--full", help="Train the base weights instead.")] = False,
) -> None:
    """Train one adapter, or the full fine-tune upper bound. Needs the model stack."""
    from engine.runtime.train import train_adapter

    try:
        path = train_adapter(settings(), name, full_finetune=full)
    except EngineError as exc:
        fail(exc)
    console.print(f"wrote [bold]{path}[/bold]")


@skills_app.command()
def propose(
    write: Annotated[bool, typer.Option("--write", help="Save each as a spec for review.")] = False,
) -> None:
    """Recurring work no skill covered, from the session index."""
    cfg = settings()
    store = SqliteSessionStore(cfg.agent.sessions.path)
    try:
        found = PatternMiner(store, cfg.agent.patterns).propose(set(skill_names(cfg.paths.data)))
    finally:
        store.close()
    if not found:
        need = cfg.agent.patterns.min_occurrences
        console.print(f"[dim]nothing recurs in {need} or more sessions yet[/dim]")
        return
    table = Table(title="proposed skills", title_justify="left")
    for column in ("skill", "seen", "sessions", "pairs", "description"):
        table.add_column(column, overflow="fold")
    for p in found:
        table.add_row(
            p.name, str(p.occurrences), str(len(p.sessions)), str(len(p.pairs)), p.description
        )
    console.print(table)
    if write:
        written, kept = write_proposals(found, cfg.proposals_dir)
        for path in written:
            console.print(f"wrote [bold]{path}[/bold]")
        for path in kept:
            console.print(f"[dim]kept {path}, already there[/dim]")
        if written:
            console.print("review each, set approved to true, then: engine skills collect <file>")


@skills_app.command()
def specs() -> None:
    """The skill specs waiting for review, and whether each is approved."""
    cfg = settings()
    found = list_specs(cfg.proposals_dir)
    if not found:
        console.print(f"[dim]no specs in {cfg.proposals_dir}[/dim]")
        return
    table = Table(title=f"skill specs in {cfg.proposals_dir}", title_justify="left")
    for column in ("file", "skill", "approved", "source", "seen", "description"):
        table.add_column(column, overflow="fold")
    for path, spec in found:
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


@skills_app.command()
def new(
    name: Annotated[str, typer.Argument(help="Lower snake case skill name.")],
    description: Annotated[str, typer.Option(help="What the skill does.")],
    instruction: Annotated[str, typer.Option(help="The instruction every input is given with.")],
) -> None:
    """Write an unapproved spec by hand. Edit it, add pairs, then set approved to true."""
    cfg = settings()
    path = cfg.proposals_dir / f"{name}.json"
    try:
        write_spec(SkillSpec(name=name, description=description, instruction=instruction), path)
    except (EngineError, ValueError) as exc:
        fail(exc)
    console.print(f"wrote [bold]{path}[/bold]; review it and set approved to true to collect")


@skills_app.command()
def collect(
    spec_path: Annotated[Path, typer.Argument(help="A spec file under the proposals directory.")],
    n: Annotated[int | None, typer.Option("--n", "-n", min=3, help="Examples to collect.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing dataset skill.")] = False,
) -> None:
    """Collect an approved spec into a dataset skill with the teacher model."""
    from engine.collect.pipeline import collect as run_collect
    from engine.collect.teacher import OpenAITeacher

    cfg = settings()
    try:
        spec = read_spec(spec_path)
        out = run_collect(cfg, spec, OpenAITeacher(cfg.collect), n=n, overwrite=overwrite)
    except EngineError as exc:
        fail(exc)
    console.print(f"wrote [bold]{out}[/bold]; next: engine skills train {spec.name}")
