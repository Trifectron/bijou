"""engine chat: talk to the agent turn after turn, in the terminal or in JSON lines.

Each turn continues the session before it, so the agent keeps the thread until /new. The process
stays up between turns, so the skill bank and MCP servers open once, and it serves the agent's
Prometheus metrics on telemetry.metrics_port for as long as it runs.

With --jsonl, one JSON object per line each way, and nothing else is written to stdout:

  in   {"op": "ask", "text": ...}  {"op": "confirm", "approve": true}  {"op": "new"}
       {"op": "stats"}
  out  {"type": "ready", "metrics": url or null}
       {"type": "event", "event": TraceEvent}   each event of a turn, as it happens
       {"type": "result", "result": RunResult}  a turn's result, without its events
       {"type": "stats", "stats": {name: number}}   every metric this agent has counted
       {"type": "error", "message": ...}        a line that could not be acted on
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

import typer

from engine.commands.agent import show_result
from engine.commands.app import app, console, err, fail, settings
from engine.commands.live import LiveTrace
from engine.core.config import Telemetry
from engine.core.protocols import TraceSink
from engine.core.types.agent import RunResult, TraceEvent
from engine.core.types.errors import ConfirmationError, EngineError
from engine.telemetry.metrics import serve_metrics, stats
from engine.wiring import Agent, open_agent

Write = Callable[[dict[str, Any]], None]


class Conversation:
    """The session the next turn continues, and the action it may be waiting on."""

    def __init__(self, agent: Agent, user_id: str) -> None:
        self.agent = agent
        self.user_id = user_id
        self.last: RunResult | None = None

    async def ask(self, text: str) -> RunResult:
        resume = self.last.session_id if self.last else None
        self.last = await self.agent.orchestrator.run(text, resume, self.user_id)
        return self.last

    async def confirm(self, approve: bool) -> RunResult:
        pending = self.last.confirmation if self.last else None
        if self.last is None or pending is None:
            raise ConfirmationError("nothing is waiting on you")
        session = self.last.session_id
        self.last = await self.agent.orchestrator.confirm(session, pending.token, approve)
        return self.last

    def new(self) -> None:
        self.last = None


class LineSink:
    """Writes each trace event as an event line. Satisfies TraceSink."""

    def __init__(self, write: Write) -> None:
        self.write = write

    def emit(self, event: TraceEvent) -> None:
        self.write({"type": "event", "event": event.model_dump(mode="json")})


async def converse(convo: Conversation, lines: AsyncIterator[str], write: Write) -> None:
    """Act on each line until the input ends. A line that cannot be acted on gets an error line."""
    async for line in lines:
        try:
            answer = await _act(convo, _message(line))
        except (EngineError, ValueError) as exc:
            write({"type": "error", "message": str(exc)})
            continue
        if answer is not None:
            write(answer)


def _message(line: str) -> dict[str, Any]:
    """One message off the wire, or ValueError naming what is wrong with the line."""
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError(f"expected an object, got {type(message).__name__}")
    return message


async def _act(convo: Conversation, message: dict[str, Any]) -> dict[str, Any] | None:
    """What to write back for one message, or None when there is nothing to say."""
    op = message.get("op")
    if op == "ask":
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ask needs text")
        return _result(await convo.ask(text))
    if op == "confirm":
        return _result(await convo.confirm(bool(message.get("approve", False))))
    if op == "new":
        convo.new()
        return None
    if op == "stats":
        return {"type": "stats", "stats": stats(convo.agent.registry)}
    raise ValueError(f"unknown op {op!r}")


def _result(result: RunResult) -> dict[str, Any]:
    """A turn's result. Its events went out as they happened, so they are left off."""
    return {"type": "result", "result": result.model_dump(mode="json", exclude={"events"})}


async def _stdin() -> AsyncIterator[str]:
    """Every non-empty line of stdin, read through the loop so no thread outlives an interrupt."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    while line := await reader.readline():
        text = line.decode(errors="replace")
        if text.strip():
            yield text


async def _terminal(convo: Conversation) -> None:
    console.print(
        "[dim]Each message continues the conversation. /new starts over, /exit leaves.[/dim]"
    )
    while True:
        try:
            # Reading on this thread blocks the loop, which has nothing else to do between turns,
            # and leaves nothing behind on Ctrl-C.
            text = console.input("[bold green]you[/bold green] › ").strip()
        except EOFError:
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/new":
            convo.new()
            console.print("[dim]new conversation[/dim]")
            continue
        if not text:
            continue
        try:
            result = await convo.ask(text)
            show_result(result)
            while (pending := result.confirmation) is not None:
                approve = typer.confirm(f"{pending.summary}\nAllow it?", default=False)
                result = await convo.confirm(approve)
                show_result(result)
        except EngineError as exc:
            err.print(f"[red]error[/red] {exc}")


def protocol_write() -> Write:
    """Write protocol lines to stdout; everything else that prints goes to stderr from here on."""
    out = sys.stdout
    sys.stdout = sys.stderr

    def write(message: dict[str, Any]) -> None:
        out.write(json.dumps(message) + "\n")
        out.flush()

    return write


def _serve_metrics(agent: Agent, telemetry: Telemetry) -> tuple[str | None, Callable[[], None]]:
    """The metrics URL and a call that stops serving it; none when metrics_port is 0."""
    if not telemetry.metrics_port:
        return None, lambda: None
    where = f"{telemetry.metrics_host}:{telemetry.metrics_port}"
    try:
        port, stop = serve_metrics(agent.registry, telemetry.metrics_host, telemetry.metrics_port)
    except OSError as exc:
        raise EngineError(
            f"telemetry.metrics_port {where} is taken ({exc}); stop the engine chat already "
            "running, or set telemetry.metrics_port = 0 to serve no metrics"
        ) from exc
    return f"http://{telemetry.metrics_host}:{port}/metrics", stop


@app.command()
def chat(
    jsonl: Annotated[
        bool, typer.Option("--jsonl", help="One JSON object per line each way, for the console.")
    ] = False,
    user: Annotated[
        str, typer.Option("--user", help="Who is asking; every session and trace carries it.")
    ] = "local",
) -> None:
    """Talk to the agent; each message continues the conversation. Serves metrics meanwhile."""
    cfg = settings()
    write = protocol_write() if jsonl else None
    # The console reads the event lines itself; the terminal has them printed as they happen.
    sinks: list[TraceSink] = [LineSink(write)] if write else []
    if write is None and cfg.agent.trace.live:
        sinks.append(LiveTrace(console))

    async def go() -> None:
        async with open_agent(cfg, sinks) as agent:
            url, stop = _serve_metrics(agent, cfg.telemetry)
            try:
                convo = Conversation(agent, user)
                if write is not None:
                    write({"type": "ready", "metrics": url})
                    await converse(convo, _stdin(), write)
                else:
                    if url:
                        console.print(f"[dim]metrics at {url}[/dim]")
                    await _terminal(convo)
            finally:
                stop()

    try:
        asyncio.run(go())
    except EngineError as exc:
        fail(exc)
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
