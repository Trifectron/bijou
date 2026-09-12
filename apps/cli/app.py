"""The console application.

Units on the left, the selected unit's output in the middle, the chat with the agent on the
right. The agent starts with the console, and the logs of every compose service that is up are
followed from the start; tasks run when asked. Keys follow vim: j and k move, enter starts or
stops, i types to the agent, colon opens the command line, slash searches. q stops every unit
and quits.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from cli.chat import Transcript, decode
from cli.core.config import Config, ConfigError
from cli.logs import LogBuffer, LogLine, LogWriter, Stream
from cli.meters import Meters
from cli.runner import Runner
from cli.status import Snapshot, snapshot
from cli.units import Command, Kind, Unit, adhoc, catalog, parse_command


class Status(StrEnum):
    """Where a unit is in its life."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    OK = "ok"
    FAILED = "failed"


class Mode(StrEnum):
    """What a key press means."""

    NORMAL = "NORMAL"
    COMMAND = "COMMAND"
    SEARCH = "SEARCH"
    CHAT = "CHAT"


class Pane(StrEnum):
    """Which pane the movement keys act on, left to right."""

    UNITS = "units"
    LOGS = "logs"
    CHAT = "chat"


PANES = list(Pane)

GLYPHS = {
    Status.IDLE: ("○", "dim"),
    Status.RUNNING: ("●", "green"),
    Status.STOPPING: ("◌", "yellow"),
    Status.OK: ("✓", "green"),
    Status.FAILED: ("✗", "red"),
}

STREAM_STYLES: dict[Stream, str] = {"out": "", "err": "", "meta": "cyan"}

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

HINTS = (
    "i chat  j/k move  ⏎ start/stop  h/l panes  m metrics  g nvtop  t htop  / search  "
    ": command  ? help  q quit"
)

HELP = """keys
  i                          type to the agent; enter sends, esc leaves
  a  d                       approve or deny the action the agent is waiting on
  R                          start a new conversation
  m                          show or hide the metrics pane
  g  t                       hand the terminal to nvtop or htop until you quit it
  j k  gg G  ctrl+d ctrl+u   move, or scroll the focused pane; G on logs resumes following
  enter  s                   start or stop the selected unit; a service row starts or stops
                             that compose service, and its logs follow while it runs
  x  r                       stop, restart
  h  l  tab                  focus units, logs, chat
  /  then n  N               search the selected unit's logs
  C                          clear the selected unit's logs
  :                          command line
  ?                          this help; any key closes it
  q                          quit; running units are stopped

commands
  :start <unit>  :stop <unit>  :restart <unit>  :clear  :help  :q
  anything else runs as a just recipe, e.g. :skills train json_extract

the agent starts with the console, and each compose service that is up has its logs followed;
x detaches a follower without stopping the service.
every line a unit prints is also appended to {log_dir}/<unit>.log
"""


@dataclass
class UnitState:
    """A unit and what the console tracks about it."""

    unit: Unit
    logs: LogBuffer
    status: Status = Status.IDLE
    code: int | None = None
    started: float | None = None
    ended: float | None = None
    scroll: int = 0
    follow: bool = True
    restart: bool = False


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _elapsed(seconds: float) -> str:
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m{whole % 60:02d}s"
    return f"{whole // 3600}h{whole % 3600 // 60:02d}m"


class ConsoleApp(App[None]):
    """The bijou developer console."""

    TITLE = "bijou"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: vertical; }
    #status, #footer { height: 1; padding: 0 1; }
    #body { height: 1fr; }
    #units { width: 34; height: 100%; border: round grey; }
    #logscol { width: 1fr; height: 100%; }
    #logs { height: 1fr; border: round grey; }
    #meters { height: 10; border: round grey; padding: 0 1; }
    #chat { width: 1fr; height: 100%; }
    #transcript { height: 1fr; border: round grey; padding: 0 1; }
    #ask { height: 3; border: round grey; padding: 0 1; }
    #units.focused, #logs.focused, #transcript.focused, #ask.focused { border: round $accent; }
    """

    def __init__(
        self,
        cfg: Config,
        root: Path,
        units: Sequence[Unit] | None = None,
        launcher: Sequence[str] = ("just",),
        probe: Callable[[Config], Snapshot] = snapshot,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = root
        chosen = catalog(cfg.eval.skills) if units is None else list(units)
        self.states = [UnitState(u, LogBuffer(cfg.console.log_lines)) for u in chosen]
        self.agent = next((s for s in self.states if s.unit.kind is Kind.AGENT), None)
        self.transcript = Transcript()
        self.meters = Meters()
        self.show_meters = True
        self.selected = 0
        self.pane = Pane.UNITS
        self.key_mode = Mode.NORMAL
        self.typed = ""
        self.draft = ""
        self.chat_scroll = 0
        self.search = ""
        self.hit: int | None = None
        self.notice = ""
        self.show_help = False
        self.pending_g = False
        self.snap: Snapshot | None = None
        self.probe = probe
        self.writer = LogWriter(root / cfg.console.log_dir)
        self.runner = Runner(root, self._on_line, self._on_exit, launcher)
        self._pending: set[asyncio.Task[None]] = set()
        self._running_services: frozenset[str] = frozenset()
        self._dirty = True
        self._handed_over = False

    # ---------- layout and lifecycle ----------

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Horizontal(id="body"):
            yield Static(id="units")
            with Vertical(id="logscol"):
                yield Static(id="logs")
                yield Static(id="meters")
            with Vertical(id="chat"):
                yield Static(id="transcript")
                yield Static(id="ask")
        yield Static(id="footer")

    async def on_mount(self) -> None:
        self.set_interval(0.1, self._refresh_if_dirty)
        self.set_interval(1.0, self._touch)
        self.set_interval(self.cfg.console.status_interval_secs, self._poll)
        self.set_interval(self.cfg.console.metrics_interval_secs, self._ask_stats)
        self.run_worker(self._poll(), exclusive=True, group="status")
        if self.agent is not None:
            await self._start(self.agent)
        self._paint()

    def on_resize(self) -> None:
        self._dirty = True

    async def on_unmount(self) -> None:
        self.runner.shutdown()
        await self.runner.drain()
        self.writer.close()

    async def _poll(self) -> None:
        self.snap = await asyncio.to_thread(self.probe, self.cfg)
        await self._follow(self.snap.running)
        self._dirty = True

    async def _follow(self, running: frozenset[str]) -> None:
        """Follow the logs of every service that has come up since the last look. A service
        followed and then stopped by hand stays stopped until it comes up again."""
        came_up = running - self._running_services
        self._running_services = running
        for tracked in self.states:
            unit = tracked.unit
            followable = unit.kind is Kind.SERVICE and unit.service in came_up
            if followable and not self.runner.owns(unit.id):
                await self._start(tracked)

    async def _ask_stats(self) -> None:
        """Ask the agent what it has counted, for the metrics pane."""
        agent = self.agent
        if agent is None or not self.show_meters or not self.transcript.ready:
            return
        if self.runner.owns(agent.unit.id):
            await self.runner.send(agent.unit.id, json.dumps({"op": "stats"}))

    def _touch(self) -> None:
        self._dirty = True

    def _refresh_if_dirty(self) -> None:
        # The spinner turns while the agent works.
        if (self._dirty or self.transcript.waiting) and not self._handed_over:
            self._dirty = False
            self._paint()

    @property
    def current(self) -> UnitState:
        return self.states[self.selected]

    def state(self, unit_id: str) -> UnitState | None:
        """The tracked state of one unit, by id."""
        return next((s for s in self.states if s.unit.id == unit_id), None)

    # ---------- runner callbacks ----------

    def _on_line(self, unit_id: str, stream: Stream, text: str) -> None:
        tracked = self.state(unit_id)
        if tracked is not None and tracked.unit.kind is Kind.AGENT and stream == "out":
            message = decode(text)
            if message is not None:
                self._dirty = True
                if message.get("type") == "stats":
                    counted = message.get("stats")
                    self.meters.update(counted if isinstance(counted, dict) else {})
                    return
                logged = self.transcript.apply(message)
                if logged is None:
                    return
                text = logged
        line = LogLine(datetime.now(), stream, text)
        if tracked is not None:
            tracked.logs.append(line)
        self.writer.append(unit_id, line)
        self._dirty = True

    def _on_exit(self, unit_id: str, code: int | None) -> None:
        tracked = self.state(unit_id)
        if tracked is None:
            return
        stopped = tracked.status is Status.STOPPING
        tracked.ended = time.monotonic()
        tracked.code = code
        if code == 0:
            tracked.status = Status.OK
        else:
            tracked.status = Status.IDLE if stopped else Status.FAILED
        self._on_line(unit_id, "meta", "stopped" if stopped else f"exited with code {code}")
        if tracked.unit.kind is Kind.AGENT:
            self.transcript.stopped(code)
        if tracked.restart:
            tracked.restart = False
            task = asyncio.get_running_loop().create_task(self._start(tracked))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    # ---------- keys ----------

    async def on_key(self, event: events.Key) -> None:
        event.stop()
        event.prevent_default()
        self.notice = ""
        if self.show_help:
            self.show_help = False
        elif self.key_mode is Mode.NORMAL:
            await self._key_normal(event)
        else:
            await self._key_line(event)
        self._dirty = True

    async def _key_normal(self, event: events.Key) -> None:
        key = event.key
        char = event.character if event.is_printable else None
        after_g = self.pending_g
        self.pending_g = False
        if char == "q":
            self._quit()
        elif char == "?":
            self.show_help = True
        elif char == "i":
            self.key_mode = Mode.CHAT
            self.pane = Pane.CHAT
        elif char in ("a", "d"):
            await self._confirm(approve=char == "a")
        elif char == "R":
            await self._new_conversation()
        elif char == "m":
            self.show_meters = not self.show_meters
        elif char == "g":
            await self._take_terminal(("nvtop",))
        elif char == "t":
            await self._take_terminal(("htop",))
        elif char == "j" or key == "down":
            self._move(1)
        elif char == "k" or key == "up":
            self._move(-1)
        elif key == "ctrl+d":
            self._move(self._rows() // 2)
        elif key == "ctrl+u":
            self._move(-(self._rows() // 2))
        elif char == "g":
            if after_g:
                self._to_top()
            else:
                self.pending_g = True
        elif char == "G":
            self._to_bottom()
        elif key == "enter" and self.pane is Pane.CHAT:
            self.key_mode = Mode.CHAT
        elif key == "enter" or char == "s":
            await self._toggle(self.current)
        elif char == "x":
            self._stop(self.current)
        elif char == "r":
            await self._restart(self.current)
        elif char == "h" or key == "left":
            self.pane = PANES[max(PANES.index(self.pane) - 1, 0)]
        elif char == "l" or key == "right":
            self.pane = PANES[min(PANES.index(self.pane) + 1, len(PANES) - 1)]
        elif key == "tab":
            self.pane = PANES[(PANES.index(self.pane) + 1) % len(PANES)]
        elif char == "C":
            await self._run(Command("clear"))
        elif char == ":":
            self.key_mode = Mode.COMMAND
            self.typed = ""
        elif char == "/":
            self.key_mode = Mode.SEARCH
            self.pane = Pane.LOGS
            self.typed = ""
        elif char in ("n", "N"):
            self._search_step(backwards=char == "N")

    async def _key_line(self, event: events.Key) -> None:
        key = event.key
        chat = self.key_mode is Mode.CHAT
        text = self.draft if chat else self.typed
        if key == "escape":
            self.key_mode = Mode.NORMAL
            return
        if key == "enter":
            if chat:
                if await self._send(text.strip()):
                    text = ""
            else:
                text = ""
                command = self.key_mode is Mode.COMMAND
                self.key_mode = Mode.NORMAL
                if command:
                    await self._run(parse_command(self.typed))
                else:
                    self.search = self.typed
                    self.hit = None
                    self._search_step(backwards=False)
        elif key == "backspace":
            text = text[:-1]
        elif key == "ctrl+u":
            text = ""
        elif event.is_printable and event.character:
            text += event.character
        if chat:
            self.draft = text
        else:
            self.typed = text

    # ---------- navigation ----------

    def _rows(self) -> int:
        return max(self.query_one("#logs", Static).content_size.height, 1)

    def _log_top(self, tracked: UnitState) -> int:
        last_top = max(len(tracked.logs) - self._rows(), 0)
        return last_top if tracked.follow else _clamp(tracked.scroll, 0, last_top)

    def _select(self, index: int) -> None:
        self.selected = _clamp(index, 0, len(self.states) - 1)
        self.hit = None

    def _move(self, n: int) -> None:
        if self.pane is Pane.UNITS:
            self._select(self.selected + n)
            return
        if self.pane is Pane.CHAT:
            # The transcript is pinned to its end; scrolling counts lines back from there.
            self.chat_scroll = max(self.chat_scroll - n, 0)
            return
        tracked = self.current
        last_top = max(len(tracked.logs) - self._rows(), 0)
        tracked.scroll = _clamp(self._log_top(tracked) + n, 0, last_top)
        tracked.follow = tracked.scroll >= last_top

    def _to_top(self) -> None:
        if self.pane is Pane.UNITS:
            self._select(0)
        elif self.pane is Pane.CHAT:
            self.chat_scroll = 1 << 30
        else:
            self.current.scroll = 0
            self.current.follow = False

    def _to_bottom(self) -> None:
        if self.pane is Pane.UNITS:
            self._select(len(self.states) - 1)
        elif self.pane is Pane.CHAT:
            self.chat_scroll = 0
        else:
            self.current.follow = True

    def _search_step(self, backwards: bool) -> None:
        if not self.search:
            self.notice = "no search yet; press /"
            return
        tracked = self.current
        step = -1 if backwards else 1
        start = self._log_top(tracked) if self.hit is None else self.hit + step
        found = tracked.logs.find(self.search, start, backwards)
        if found is None:
            self.notice = f"no match for {self.search!r}"
            return
        self.hit = found
        rows = self._rows()
        tracked.follow = False
        tracked.scroll = _clamp(found - rows // 2, 0, max(len(tracked.logs) - rows, 0))
        self.notice = f"/{self.search}  line {found + 1} of {len(tracked.logs)}"

    # ---------- the agent ----------

    def _agent_up(self) -> bool:
        return self.agent is not None and self.runner.owns(self.agent.unit.id)

    async def _tell_agent(self, message: dict[str, Any]) -> bool:
        """Send one message to the agent process. False when it did not take it."""
        if self.agent is None:
            self.notice = "there is no agent unit"
            return False
        if await self.runner.send(self.agent.unit.id, json.dumps(message)):
            return True
        self.notice = "the agent is not running; select it and press enter"
        return False

    async def _send(self, text: str) -> bool:
        """Ask the agent; the reply continues the conversation. False when nothing was sent."""
        if not text:
            return False
        if not self._agent_up() or not self.transcript.ready:
            self.notice = "the agent is not ready; select it to see why"
            return False
        if self.transcript.waiting:
            self.notice = "the agent is still on the last message"
            return False
        if self.transcript.pending:
            self.notice = "the agent is holding an action: a approves it, d denies it"
            return False
        if not await self._tell_agent({"op": "ask", "text": text}):
            return False
        self.transcript.ask(text)
        self.chat_scroll = 0
        if self.agent is not None:
            # The middle pane shows the agent's events while it works.
            self._select(self.states.index(self.agent))
        return True

    async def _confirm(self, approve: bool) -> None:
        if self.transcript.pending is None:
            self.notice = "nothing is waiting on you"
            return
        if await self._tell_agent({"op": "confirm", "approve": approve}):
            self.transcript.confirming(approve)

    async def _new_conversation(self) -> None:
        if self.transcript.waiting:
            self.notice = "the agent is still on the last message"
            return
        if self._agent_up() and not await self._tell_agent({"op": "new"}):
            return
        self.transcript.new()
        self.chat_scroll = 0
        self.notice = "new conversation"

    # ---------- control ----------

    async def _toggle(self, tracked: UnitState) -> None:
        if tracked.unit.kind is Kind.SERVICE:
            await self._service(tracked)
        elif self.runner.owns(tracked.unit.id):
            self._stop(tracked)
        else:
            await self._start(tracked)

    async def _service(self, tracked: UnitState) -> None:
        """Start or stop the compose service this unit is. Its logs follow while it runs."""
        service = tracked.unit.service
        up = self.snap is not None and service in self.snap.running
        if up and self.runner.owns(tracked.unit.id):
            self.runner.stop(tracked.unit.id)
        await self.runner.run_once(tracked.unit.id, ("stop" if up else "up", service))
        self.notice = f"{'stopping' if up else 'starting'} {tracked.unit.name}"

    async def _start(self, tracked: UnitState) -> None:
        unit_id = tracked.unit.id
        if self.runner.owns(unit_id):
            self.notice = f"{tracked.unit.name} is already running"
            return
        tracked.status = Status.RUNNING
        tracked.code = None
        tracked.started = time.monotonic()
        tracked.ended = None
        tracked.follow = True
        agent = tracked.unit.kind is Kind.AGENT
        if agent:
            self.transcript.restarted()
        try:
            await self.runner.start(unit_id, tracked.unit.args, stdin=agent)
        except OSError as exc:
            self._failed_to_start(tracked, exc)
            return
        self.notice = f"started {tracked.unit.name}"
        self._dirty = True

    async def _take_terminal(self, args: Sequence[str]) -> None:
        """Hand the whole terminal to a full-screen tool until it exits.

        Painting is held meanwhile; the runner's tasks keep reading every unit.
        """
        cmd = [*self.runner.launcher, *args]
        try:
            with self.suspend():
                self._handed_over = True
                outcome = await self._foreground(cmd)
        except SuspendNotSupported:
            self.notice = f"{args[0]} needs a terminal the console can hand over"
            return
        finally:
            self._handed_over = False
            self._dirty = True
        if isinstance(outcome, OSError):
            self.notice = f"could not start {args[0]}: {outcome}"

    async def _foreground(self, cmd: Sequence[str]) -> int | OSError:
        """Run cmd attached to the terminal. A spawn error is returned rather than raised, so the
        suspended console is always resumed."""
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=self.root)
        except OSError as exc:
            return exc
        return await proc.wait()

    def _failed_to_start(self, tracked: UnitState, exc: OSError) -> None:
        tracked.status = Status.FAILED
        tracked.ended = time.monotonic()
        self._on_line(tracked.unit.id, "meta", f"could not start: {exc}")
        self.notice = f"could not start {tracked.unit.name}: {exc}"

    def _stop(self, tracked: UnitState) -> None:
        if self.runner.stop(tracked.unit.id):
            tracked.status = Status.STOPPING
            self.notice = f"stopping {tracked.unit.name}"
        else:
            self.notice = f"{tracked.unit.name} is not running"

    async def _restart(self, tracked: UnitState) -> None:
        if self.runner.owns(tracked.unit.id):
            tracked.restart = True
            self._stop(tracked)
        else:
            await self._start(tracked)

    async def _run(self, command: Command) -> None:
        verb = command.verb
        if verb == "quit":
            self._quit()
        elif verb == "help":
            self.show_help = True
        elif verb == "clear":
            self.current.logs.clear()
            self.current.scroll = 0
            self.current.follow = True
            self.hit = None
        elif verb in ("start", "stop", "restart"):
            tracked = self.state(command.arg)
            if tracked is None:
                self.notice = f"no unit {command.arg!r}"
                return
            self._select(self.states.index(tracked))
            if verb == "start":
                await self._start(tracked)
            elif verb == "stop":
                self._stop(tracked)
            else:
                await self._restart(tracked)
        elif verb == "just":
            await self._run_adhoc(command.args)

    async def _run_adhoc(self, args: Sequence[str]) -> None:
        if not args:
            self.notice = "usage: :just <recipe> [args]"
            return
        unit = adhoc(args)
        tracked = self.state(unit.id)
        if tracked is None:
            tracked = UnitState(unit, LogBuffer(self.cfg.console.log_lines))
            self.states.append(tracked)
        self._select(self.states.index(tracked))
        self.pane = Pane.LOGS
        await self._start(tracked)

    def _quit(self) -> None:
        self.runner.shutdown()
        self.exit()

    # ---------- drawing ----------

    def _paint(self) -> None:
        units = self.query_one("#units", Static)
        logs = self.query_one("#logs", Static)
        meters = self.query_one("#meters", Static)
        transcript = self.query_one("#transcript", Static)
        ask = self.query_one("#ask", Static)
        units.set_class(self.pane is Pane.UNITS, "focused")
        logs.set_class(self.pane is Pane.LOGS, "focused")
        transcript.set_class(self.pane is Pane.CHAT, "focused")
        ask.set_class(self.key_mode is Mode.CHAT, "focused")
        self.query_one("#status", Static).update(self._status_text())
        units.border_title = "units"
        units.update(self._units_text(max(units.content_size.height, 1)))
        self._paint_logs(logs)
        self._paint_meters(meters)
        self._paint_chat(transcript, ask)
        self.query_one("#footer", Static).update(self._footer_text())

    def _status_text(self) -> Text:
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(" bijou ", "bold reverse")
        mode_style = {
            Mode.NORMAL: "bold",
            Mode.COMMAND: "bold yellow",
            Mode.SEARCH: "bold cyan",
            Mode.CHAT: "bold green",
        }
        text.append(f" {self.key_mode} ", mode_style[self.key_mode])
        if self.agent is not None:
            up = self._agent_up()
            style = "green" if up and self.transcript.ready else "yellow" if up else "dim"
            text.append("  ● " if up else "  ○ ", style)
            text.append("agent", "" if up else "dim")
        snap = self.snap
        if snap is None:
            text.append("  reading status...", "dim")
        else:
            # The services have a light each in the sidebar, and the GPU a row in the metrics pane.
            if not snap.checkpoint:
                text.append("  ● random weights", "yellow")
            elif snap.checkpoint_present:
                text.append("  ● ", "green")
                text.append(snap.checkpoint)
            else:
                text.append(f"  ○ {snap.checkpoint} missing", "red")
            text.append(f"  adapters {snap.adapters}/{snap.skills}")
            text.append(f"  full {snap.full_finetunes}/{snap.skills}")
            if snap.last_run:
                text.append(f"  last {snap.last_run}", "dim")
            text.append(f"  git {snap.git}", "dim")
        if self.notice:
            text.append(f"   {self.notice}", "bold")
        return text

    def _units_text(self, height: int) -> Text:
        rows: list[tuple[Text, int | None]] = []
        group = None
        for i, tracked in enumerate(self.states):
            if tracked.unit.group != group:
                group = tracked.unit.group
                rows.append((Text(str(group), style="bold dim"), None))
            glyph, style = GLYPHS[tracked.status]
            if tracked.unit.kind is Kind.SERVICE:
                # A service's light is the service itself, not its log follower.
                up = self.snap is not None and tracked.unit.service in self.snap.running
                glyph, style = ("●", "green") if up else ("○", "dim")
            name_style = "reverse" if i == self.selected else ""
            rows.append((Text.assemble((f" {glyph} ", style), (tracked.unit.name, name_style)), i))
        selected_row = next(r for r, (_, i) in enumerate(rows) if i == self.selected)
        top = _clamp(selected_row - height // 2, 0, max(len(rows) - height, 0))
        out = Text("\n").join(line for line, _ in rows[top : top + height])
        out.no_wrap = True
        out.overflow = "ellipsis"
        return out

    def _paint_logs(self, logs: Static) -> None:
        tracked = self.current
        if self.show_help:
            logs.border_title = "help"
            logs.border_subtitle = "any key closes"
            logs.update(Text(HELP.format(log_dir=self.cfg.console.log_dir)))
            return
        now = time.monotonic()
        parts = [tracked.unit.name, str(tracked.status)]
        if tracked.status is Status.FAILED and tracked.code is not None:
            parts[-1] = f"failed ({tracked.code})"
        if tracked.started is not None:
            parts.append(_elapsed((tracked.ended or now) - tracked.started))
        parts += [f"{len(tracked.logs)} lines", "follow" if tracked.follow else "scroll"]
        logs.border_title = " · ".join(parts)
        logs.border_subtitle = tracked.unit.hint
        rows = self._rows()
        top = self._log_top(tracked)
        body = Text(no_wrap=True, overflow="ellipsis")
        for index, line in enumerate(tracked.logs.window(top, rows), start=top):
            if index > top:
                body.append("\n")
            row = Text.assemble((f"{line.at:%H:%M:%S} ", "dim"))
            row.append_text(Text.from_ansi(line.text, style=STREAM_STYLES[line.stream]))
            if self.search:
                row.highlight_words([self.search], "black on yellow", case_sensitive=False)
            if index == self.hit:
                row.stylize("reverse")
            body.append_text(row)
        if not len(tracked.logs):
            body = Text("press enter to start this unit, ? for help", style="dim")
        logs.update(body)

    def _paint_meters(self, meters: Static) -> None:
        meters.display = self.show_meters
        if not self.show_meters:
            return
        meters.border_title = "metrics · this agent, since it started"
        meters.border_subtitle = "m hides · g nvtop · t htop"
        width = max(meters.content_size.width, 20)
        meters.update(self.meters.render(width, self._machine_rows()))

    def _machine_rows(self) -> list[str]:
        """The GPU and the box, the numbers nvtop and htop are opened for."""
        snap = self.snap
        if snap is None:
            return []
        rows = []
        for gpu in snap.gpus:
            used, total = gpu.used_mib / 1024, gpu.total_mib / 1024
            rows.append(f"gpu   {gpu.util:>3}%  {used:.1f}/{total:.1f}G  {gpu.temp_c}°C")
            if gpu.holders:
                rows.append(f"      {', '.join(dict.fromkeys(gpu.holders))}")
        if not snap.gpus:
            rows.append("gpu   none")
        if snap.machine is not None:
            box = snap.machine
            rows.append(f"load  {box.load:.2f} over {box.cpus} cpus")
            rows.append(f"mem   {box.mem_used_gib:.1f}/{box.mem_total_gib:.1f}G")
        return rows

    def _paint_chat(self, transcript: Static, ask: Static) -> None:
        t = self.transcript
        transcript.border_title = f"chat · {t.session}" if t.session else "chat · new conversation"
        if self.agent is None:
            transcript.border_subtitle = "no agent unit"
        elif not self._agent_up():
            transcript.border_subtitle = "agent stopped"
        else:
            transcript.border_subtitle = "" if t.ready else "agent starting"
        width = max(transcript.content_size.width, 10)
        height = max(transcript.content_size.height, 1)
        frame = SPINNER[int(time.monotonic() * 10) % len(SPINNER)]
        lines = list(t.render(frame).wrap(self.console, width))
        self.chat_scroll = min(self.chat_scroll, max(len(lines) - height, 0))
        end = len(lines) - self.chat_scroll
        shown = Text("\n").join(lines[max(end - height, 0) : end])
        if not t.turns and not t.waiting:
            shown = Text("press i to talk to the agent; each message continues the last", "dim")
        transcript.update(shown)
        ask.border_title = "ask"
        if self.key_mode is Mode.CHAT:
            ask.update(Text.assemble(("> ", "bold green"), self.draft, "█"))
        elif t.pending:
            ask.update(Text("a approve · d deny · i to type", style="bold yellow"))
        else:
            ask.update(Text("i to type · enter sends · esc leaves · R new conversation", "dim"))

    def _footer_text(self) -> Text:
        if self.key_mode is Mode.COMMAND:
            return Text(f":{self.typed}█")
        if self.key_mode is Mode.SEARCH:
            return Text(f"/{self.typed}█")
        if self.key_mode is Mode.CHAT:
            return Text("chat: enter sends · esc leaves · ctrl+u clears", style="dim")
        return Text(HINTS, style="dim", no_wrap=True, overflow="ellipsis")


def repo_root(start: Path) -> Path | None:
    """The nearest directory at or above start holding the justfile and bijou.toml."""
    for directory in (start, *start.parents):
        if (directory / "justfile").exists() and (directory / "bijou.toml").exists():
            return directory
    return None


def run() -> None:
    """Start the console from anywhere inside the repo."""
    root = repo_root(Path.cwd())
    if root is None:
        raise ConfigError("run the console from inside the bijou repo; no justfile found above")
    os.chdir(root)
    ConsoleApp(Config(), root).run()
