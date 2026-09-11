"""The engine command's root. The model and agent command modules register onto it."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from engine import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "A masked diffusion LM with a bank of LoRA skills, and the agent that equips them.\n\n"
        "[bold]engine run[/bold] sends a request through the agent. [bold]engine skill list[/bold]"
        " and [bold]engine config[/bold] need no GPU."
    ),
)


def _version(value: bool) -> None:
    if value:
        Console().print(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version.")
    ] = False,
) -> None:
    """Root options."""
