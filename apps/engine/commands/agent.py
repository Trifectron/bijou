"""Commands for the agent: run a request, confirm an action, serve, and inspect what it did."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from engine.commands.app import app
from engine.core.config import AgentConfig, Config, load
from engine.core.types.agent import RunResult, RunStatus, SessionSummary
from engine.core.types.errors import EngineError, SkillRuntimeError
from engine.patterns.miner import PatternMiner, write_proposals
from engine.stores.sessions import SqliteSessionStore
from engine.wiring import open_harness, skill_runtime

console = Console()
err = Console(stderr=True)

sessions_app = typer.Typer(no_args_is_help=True, help="Every session, indexed.")
app.add_typer(sessions_app, name="sessions")


def _full() -> Config:
    try:
        return load()
    except (EngineError, ValueError) as exc:
        err.print(f"[red]error[/red] {exc}")
        raise typer.Exit(1) from exc


def _config() -> AgentConfig:
    return _full().agent


def _fail(exc: Exception) -> None:
    err.print(f"[red]error[/red] {exc}")
    raise typer.Exit(1)


STATUS_STYLE = {
    RunStatus.ANSWERED: "green",
    RunStatus.AWAITING_CONFIRMATION: "yellow",
}


def _show(result: RunResult) -> None:
    plan = result.plan
    title = f"session {result.session_id}"
    if plan.fallback:
        title += f" [yellow](planning fell back: {plan.reason})[/yellow]"
    table = Table(title=title, title_justify="left")
    for column in ("step", "goal", "status", "skills", "tools", "turns"):
        table.add_column(column, overflow="fold")
    for step in result.steps:
        style = STATUS_STYLE.get(step.status, "red")
        table.add_row(
            step.id,
            step.goal,
            f"[{style}]{step.status.value}[/{style}]",
            ", ".join(step.skills) or "[dim]none[/dim]",
            ", ".join(dict.fromkeys(step.tools)) or "[dim]none[/dim]",
            str(step.turns),
        )
    console.print(table)
    style = STATUS_STYLE.get(result.status, "red")
    console.print(
        Panel(
            result.answer or "[dim]no answer[/dim]",
            title=f"[{style}]{result.status.value}[/{style}]",
            title_align="left",
            subtitle=(
                f"{result.usage.prompt_tokens}+{result.usage.completion_tokens} tokens · "
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
) -> None:
    """Plan, equip skills, act and answer. Asks before any action that needs confirmation."""
    cfg = _full()

    async def go() -> None:
        async with open_harness(cfg) as harness:
            result = await harness.orchestrator.run(request, resume)
            while True:
                if as_json:
                    typer.echo(result.model_dump_json(indent=2))
                else:
                    _show(result)
                pending = result.confirmation
                if pending is None or as_json:
                    return
                approve = typer.confirm(f"{pending.summary}\nAllow it?", default=False)
                result = await harness.orchestrator.confirm(
                    result.session_id, pending.token, approve
                )

    try:
        asyncio.run(go())
    except EngineError as exc:
        _fail(exc)


@app.command()
def confirm(
    session_id: Annotated[str, typer.Argument(help="The session waiting on you.")],
    token: Annotated[str, typer.Argument(help="The confirmation token from the result.")],
    deny: Annotated[bool, typer.Option("--deny", help="Decline instead of approving.")] = False,
) -> None:
    """Approve or decline the action a session is waiting on, and carry it on."""
    cfg = _full()

    async def go() -> RunResult:
        async with open_harness(cfg) as harness:
            return await harness.orchestrator.confirm(session_id, token, not deny)

    try:
        _show(asyncio.run(go()))
    except EngineError as exc:
        _fail(exc)


@app.command()
def serve() -> None:
    """Serve the agent over HTTP on agent.http."""
    import uvicorn

    from engine.routes.agent import create_app

    cfg = _full()
    uvicorn.run(create_app(cfg), host=cfg.agent.http.host, port=cfg.agent.http.port)


@app.command()
def skills() -> None:
    """What the skill bank can equip, in this process or at agent.skills.url."""
    cfg = _full()
    where = "this process" if cfg.agent.skills.mode == "local" else cfg.agent.skills.url

    async def go() -> None:
        runtime = skill_runtime(cfg)
        try:
            listed = await runtime.catalog()
        finally:
            await runtime.aclose()
        table = Table(title=f"skills in {where}", title_justify="left")
        for column in ("skill", "trained", "description"):
            table.add_column(column, overflow="fold")
        for skill in listed:
            mark = "[green]yes[/green]" if skill.trained else "[dim]no[/dim]"
            table.add_row(skill.name, mark, skill.description)
        console.print(table)

    try:
        asyncio.run(go())
    except SkillRuntimeError as exc:
        _fail(exc)


@app.command()
def patterns(
    write: Annotated[
        bool, typer.Option("--write", help="Write each proposal as a spec for review.")
    ] = False,
) -> None:
    """Recurring work no skill covered, proposed as new skills."""
    full = _full()
    cfg = full.agent
    store = SqliteSessionStore(cfg.sessions.path)

    async def existing() -> set[str]:
        runtime = skill_runtime(full)
        try:
            return {s.name for s in await runtime.catalog()}
        except SkillRuntimeError:
            err.print("[yellow]skill bank unavailable; names are not checked against it[/yellow]")
            return set()
        finally:
            await runtime.aclose()

    proposals = PatternMiner(store, cfg.patterns).propose(asyncio.run(existing()))
    store.close()
    if not proposals:
        console.print(
            f"[dim]nothing recurs in {cfg.patterns.min_occurrences} or more sessions yet[/dim]"
        )
        return
    table = Table(title="proposed skills", title_justify="left")
    for column in ("skill", "seen", "sessions", "pairs", "description"):
        table.add_column(column, overflow="fold")
    for p in proposals:
        table.add_row(
            p.name, str(p.occurrences), str(len(p.sessions)), str(len(p.pairs)), p.description
        )
    console.print(table)
    if write:
        written, skipped = write_proposals(proposals, cfg.patterns.proposals_dir)
        for path in written:
            console.print(f"wrote [bold]{path}[/bold]")
        for path in skipped:
            console.print(f"[dim]kept {path}, already there[/dim]")
        if written:
            console.print("review each, set approved to true, then: just collect run <file>")


@sessions_app.command("list")
def sessions_list(
    limit: Annotated[int, typer.Option(min=1, help="Show the newest N.")] = 20,
) -> None:
    """Every session, newest first."""
    cfg = _config()
    store = SqliteSessionStore(cfg.sessions.path)
    _summaries(store.recent(limit), f"sessions in {cfg.sessions.path}")
    store.close()


@sessions_app.command("search")
def sessions_search(
    query: Annotated[str, typer.Argument(help="Words to find in requests, answers and steps.")],
    limit: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Sessions matching every word of the query."""
    cfg = _config()
    store = SqliteSessionStore(cfg.sessions.path)
    _summaries(store.search(query, limit), f"sessions matching {query!r}")
    store.close()


@sessions_app.command("show")
def sessions_show(session_id: Annotated[str, typer.Argument(help="A session id.")]) -> None:
    """One session in full, as stored."""
    cfg = _config()
    store = SqliteSessionStore(cfg.sessions.path)
    record = store.get(session_id)
    store.close()
    if record is None:
        err.print(f"[red]error[/red] no session {session_id}")
        raise typer.Exit(1)
    typer.echo(record.model_dump_json(indent=2))


def _summaries(rows: list[SessionSummary], title: str) -> None:
    if not rows:
        console.print("[dim]no sessions[/dim]")
        return
    table = Table(title=title, title_justify="left")
    for column in ("session", "started", "status", "request", "answer"):
        table.add_column(column, overflow="fold")
    for s in rows:
        style = STATUS_STYLE.get(s.status, "red")
        table.add_row(
            s.id,
            f"{s.created_at:%Y-%m-%d %H:%M}",
            f"[{style}]{s.status.value}[/{style}]",
            s.request,
            s.answer,
        )
    console.print(table)


if __name__ == "__main__":
    app()
