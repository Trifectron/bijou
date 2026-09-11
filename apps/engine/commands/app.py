"""The engine command's root, and what every command module shares."""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer
from rich.console import Console

from engine import __version__
from engine.core.config import Config, load
from engine.core.types.errors import EngineError

console = Console()
err = Console(stderr=True)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "A masked diffusion LM with a bank of LoRA skills, and the agent that equips them.\n\n"
        "[bold]engine run[/bold] sends a request through the agent. "
        "[bold]engine skills list[/bold] and [bold]engine config[/bold] need no GPU."
    ),
)


def fail(exc: Exception) -> NoReturn:
    """Report a known failure without a traceback and exit 1."""
    err.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


def settings() -> Config:
    """The configuration, or a clean exit naming what is wrong with it."""
    try:
        return load()
    except (EngineError, ValueError) as exc:
        fail(exc)


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
