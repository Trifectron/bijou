"""Commands for the agent: run a request, confirm an action, and look through sessions."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from engine.commands.app import app, console, fail, settings
from engine.core.types.agent import RunResult, RunStatus, SessionSummary
from engine.core.types.errors import EngineError
from engine.memory.sessions import SqliteSessionStore
from engine.wiring import open_agent

STATUS_STYLE = {RunStatus.ANSWERED: "green", RunStatus.AWAITING_CONFIRMATION: "yellow"}


def _status(status: RunStatus) -> str:
    style = STATUS_STYLE.get(status, "red")
    return f"[{style}]{status.value}[/{style}]"


def show_result(result: RunResult) -> None:
    plan = result.plan
    title = f"session {result.session_id}"
    if plan.fallback:
        title += f" [yellow](planning fell back: {plan.reason})[/yellow]"
    table = Table(title=title, title_justify="left")
    for column in ("step", "goal", "status", "skills", "tools", "turns"):
        table.add_column(column, overflow="fold")
    for step in result.steps:
        table.add_row(
            step.id,
            step.goal,
            _status(step.status),
            ", ".join(step.skills) or "[dim]none[/dim]",
            ", ".join(dict.fromkeys(step.tools)) or "[dim]none[/dim]",
            str(step.turns),
        )
    console.print(table)
    console.print(
        Panel(
            result.answer or "[dim]no answer[/dim]",
            title=_status(result.status),
            title_align="left",
            subtitle=(
                f"{result.usage.prompt_tokens}+{result.usage.completion_tokens} tokens, "
                f"{result.duration_ms / 1000:.1f}s"
            ),
            subtitle_align="right",
        )
    )


@app.command()
def run(
    request: Annotated[str, typer.Argument(help="What you want done.")],
    resume: Annotated[
        str | None, typer.Option("--resume", help="Continue from an earlier session.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    user: Annotated[
        str, typer.Option("--user", help="Who is asking; the session and its trace carry it.")
    ] = "local",
) -> None:
    """Plan, equip skills, act and answer. Asks before any action that needs confirmation."""
    cfg = settings()

    async def go() -> None:
        async with open_agent(cfg) as agent:
            result = await agent.orchestrator.run(request, resume, user)
            while True:
                if as_json:
                    typer.echo(result.model_dump_json(indent=2))
                else:
                    show_result(result)
                pending = result.confirmation
                if pending is None or as_json:
                    return
                approve = typer.confirm(f"{pending.summary}\nAllow it?", default=False)
                result = await agent.orchestrator.confirm(result.session_id, pending.token, approve)

    try:
        asyncio.run(go())
    except EngineError as exc:
        fail(exc)


@app.command()
def confirm(
    session_id: Annotated[str, typer.Argument(help="The session waiting on you.")],
    token: Annotated[str, typer.Argument(help="The confirmation token from the result.")],
    deny: Annotated[bool, typer.Option("--deny", help="Decline instead of approving.")] = False,
) -> None:
    """Approve or decline the action a session is waiting on, and carry it on."""
    cfg = settings()

    async def go() -> RunResult:
        async with open_agent(cfg) as agent:
            return await agent.orchestrator.confirm(session_id, token, not deny)

    try:
        show_result(asyncio.run(go()))
    except EngineError as exc:
        fail(exc)


@app.command()
def sessions(
    query: Annotated[
        str | None, typer.Argument(help="Words to find in requests, answers and steps.")
    ] = None,
    show: Annotated[str | None, typer.Option("--show", help="Print one session in full.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="At most this many.")] = 20,
) -> None:
    """Every session, newest first, or those matching every word of a query."""
    cfg = settings().agent
    store = SqliteSessionStore(cfg.sessions.path)
    try:
        if show is not None:
            record = store.get(show)
            if record is None:
                fail(KeyError(f"no session {show}"))
            typer.echo(record.model_dump_json(indent=2))
            return
        rows = store.search(query, limit) if query else store.recent(limit)
    finally:
        store.close()
    title = f"sessions matching {query!r}" if query else f"sessions in {cfg.sessions.path}"
    _summaries(rows, title)


def _summaries(rows: list[SessionSummary], title: str) -> None:
    if not rows:
        console.print("[dim]no sessions[/dim]")
        return
    table = Table(title=title, title_justify="left")
    for column in ("session", "started", "status", "request", "answer"):
        table.add_column(column, overflow="fold")
    for s in rows:
        table.add_row(
            s.id, f"{s.created_at:%Y-%m-%d %H:%M}", _status(s.status), s.request, s.answer
        )
    console.print(table)
