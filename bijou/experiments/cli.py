"""The bijou command. Every experiment enters here."""

from __future__ import annotations

import typer
from rich import print as rprint
from rich.table import Table

from bijou.core.config import load
from bijou.core.runs import load_records
from bijou.core.types import SkillReport
from bijou.skills import KNOWN

app = typer.Typer(no_args_is_help=True, add_completion=False)
skill_app = typer.Typer(no_args_is_help=True, help="Inspect and train skills.")
run_app = typer.Typer(no_args_is_help=True, help="Run records.")
app.add_typer(skill_app, name="skill")
app.add_typer(run_app, name="run")


@app.command()
def config() -> None:
    """Print the resolved configuration."""
    rprint(load().model_dump())


@skill_app.command("list")
def skill_list() -> None:
    """Every known skill and the size of its generated splits."""
    cfg = load()
    table = Table(title="skills")
    table.add_column("skill")
    table.add_column("train", justify="right")
    table.add_column("eval", justify="right")
    for name in KNOWN:
        table.add_row(name, str(cfg.train.train_samples), str(cfg.eval.eval_samples))
    rprint(table)


@skill_app.command("names")
def skill_names() -> None:
    """One skill name per line, for scripting."""
    for name in KNOWN:
        typer.echo(name)


@skill_app.command("sample")
def skill_sample(name: str, n: int = 3) -> None:
    """Show generated samples for one skill without loading a model."""
    from bijou.skills import load as load_skill

    for sample in load_skill(name).generate(n, load().eval.seed):
        rprint(f"[bold]{sample.id}[/bold]\n{sample.prompt}\n-> {sample.target}\n")


@skill_app.command("train")
def skill_train(name: str, full_finetune: bool = False) -> None:
    """Train one adapter, or the full-finetune upper bound."""
    from bijou.runtime.train import train_adapter

    path = train_adapter(load(), name, full_finetune=full_finetune)
    rprint(f"wrote {path}")


@app.command()
def evaluate() -> None:
    """Score the composition matrix."""
    from bijou.experiments import matrix

    _print(matrix.run(load()))


@run_app.command("list")
def run_list() -> None:
    """Every run record, oldest first."""
    cfg = load()
    table = Table(title=f"runs in {cfg.paths.runs}")
    for column in ("run", "kind", "git", "scores"):
        table.add_column(column)
    for record in load_records(cfg.paths.runs):
        table.add_row(
            record.run_id, record.kind, record.env.get("git_sha", "?"), str(len(record.scores))
        )
    rprint(table)


def _print(reports: list[SkillReport]) -> None:
    skills = sorted({r.skill for r in reports})
    table = Table(title="composition matrix (pass rate)")
    table.add_column("condition")
    for skill in skills:
        table.add_column(skill, justify="right")
    by_condition: dict[str, dict[str, SkillReport]] = {}
    for report in reports:
        by_condition.setdefault(report.condition, {})[report.skill] = report
    for condition, row in by_condition.items():
        table.add_row(condition, *(f"{row[s].rate:.0%}" if s in row else "-" for s in skills))
    rprint(table)


if __name__ == "__main__":
    app()
