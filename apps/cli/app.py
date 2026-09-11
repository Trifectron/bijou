"""The console application.

Units are listed on the left and the selected unit's output on the right. Keys
follow vim: j and k move, enter starts or stops, colon opens the command line,
slash searches. q stops every running unit and quits.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from cli.core.config import Config, ConfigError
from cli.logs import LogBuffer, LogLine, LogWriter, Stream
from cli.runner import Runner
from cli.status import Snapshot, snapshot
from cli.units import Command, Unit, adhoc, catalog, parse_command


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


class Pane(StrEnum):
    """Which side the movement keys act on."""

    UNITS = "units"
    LOGS = "logs"


GLYPHS = {
    Status.IDLE: ("○", "dim"),
    Status.RUNNING: ("●", "green"),
    Status.STOPPING: ("◌", "yellow"),
    Status.OK: ("✓", "green"),
    Status.FAILED: ("✗", "red"),
}

STREAM_STYLES: dict[Stream, str] = {"out": "", "err": "", "meta": "cyan"}

HINTS = (
    "j/k move  ⏎ start/stop  x stop  r restart  h/l units/logs  / search  : command  ? help  q quit"
)

HELP = """keys
  j k  gg G  ctrl+d ctrl+u   move, or scroll the focused pane; G on logs resumes following
  enter  s                   start or stop the selected unit
  x  r                       stop, restart
  h  l  tab                  focus units, logs
  /  then n  N               search the selected unit's logs
  C                          clear the selected unit's logs
  :                          command line
  ?                          this help; any key closes it
  q                          quit; running units are stopped

commands
  :start <unit>  :stop <unit>  :restart <unit>  :clear  :help  :q
  anything else runs as a just recipe, e.g. :skill sample json_extract -n 2

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
    #units { width: 46; height: 100%; border: round grey; }
    #logs { width: 1fr; height: 100%; border: round grey; }
    #units.focused, #logs.focused { border: round $accent; }
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
        self.selected = 0
        self.pane = Pane.UNITS
        self.key_mode = Mode.NORMAL
        self.typed = ""
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
        self._dirty = True

    # ---------- layout and lifecycle ----------

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        with Horizontal(id="body"):
            yield Static(id="units")
            yield Static(id="logs")
        yield Static(id="footer")

    async def on_mount(self) -> None:
        self.set_interval(0.1, self._refresh_if_dirty)
        self.set_interval(1.0, self._touch)
        self.set_interval(self.cfg.console.status_interval_secs, self._poll)
        self.run_worker(self._poll(), exclusive=True, group="status")
        self._paint()

    def on_resize(self) -> None:
        self._dirty = True

    def on_unmount(self) -> None:
        self.runner.shutdown()
        self.writer.close()

    async def _poll(self) -> None:
        self.snap = await asyncio.to_thread(self.probe, self.cfg)
        self._dirty = True

    def _touch(self) -> None:
        self._dirty = True

    def _refresh_if_dirty(self) -> None:
        if self._dirty:
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
        line = LogLine(datetime.now(), stream, text)
        tracked = self.state(unit_id)
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
        elif key == "enter" or char == "s":
            await self._toggle(self.current)
        elif char == "x":
            self._stop(self.current)
        elif char == "r":
            await self._restart(self.current)
        elif char == "h" or key == "left":
            self.pane = Pane.UNITS
        elif char == "l" or key == "right":
            self.pane = Pane.LOGS
        elif key == "tab":
            self.pane = Pane.LOGS if self.pane is Pane.UNITS else Pane.UNITS
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
        if key == "escape":
            self.key_mode = Mode.NORMAL
        elif key == "enter":
            text, self.typed = self.typed, ""
            command = self.key_mode is Mode.COMMAND
            self.key_mode = Mode.NORMAL
            if command:
                await self._run(parse_command(text))
            else:
                self.search = text
                self.hit = None
                self._search_step(backwards=False)
        elif key == "backspace":
            self.typed = self.typed[:-1]
        elif key == "ctrl+u":
            self.typed = ""
        elif event.is_printable and event.character:
            self.typed += event.character

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
        tracked = self.current
        last_top = max(len(tracked.logs) - self._rows(), 0)
        tracked.scroll = _clamp(self._log_top(tracked) + n, 0, last_top)
        tracked.follow = tracked.scroll >= last_top

    def _to_top(self) -> None:
        if self.pane is Pane.UNITS:
            self._select(0)
        else:
            self.current.scroll = 0
            self.current.follow = False

    def _to_bottom(self) -> None:
        if self.pane is Pane.UNITS:
            self._select(len(self.states) - 1)
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

    # ---------- control ----------

    async def _toggle(self, tracked: UnitState) -> None:
        if self.runner.owns(tracked.unit.id):
            self._stop(tracked)
        else:
            await self._start(tracked)

    async def _start(self, tracked: UnitState) -> None:
        unit_id = tracked.unit.id
        if self.runner.owns(unit_id):
            self.notice = f"{unit_id} is already running"
            return
        tracked.status = Status.RUNNING
        tracked.code = None
        tracked.started = time.monotonic()
        tracked.ended = None
        tracked.follow = True
        try:
            await self.runner.start(unit_id, tracked.unit.args)
        except OSError as exc:
            tracked.status = Status.FAILED
            tracked.ended = time.monotonic()
            self._on_line(unit_id, "meta", f"could not start: {exc}")
            self.notice = f"could not start {unit_id}: {exc}"
            return
        self.notice = f"started {unit_id}"
        self._dirty = True

    def _stop(self, tracked: UnitState) -> None:
        if self.runner.stop(tracked.unit.id):
            tracked.status = Status.STOPPING
            self.notice = f"stopping {tracked.unit.id}"
        else:
            self.notice = f"{tracked.unit.id} is not running"

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
        units.set_class(self.pane is Pane.UNITS, "focused")
        logs.set_class(self.pane is Pane.LOGS, "focused")
        self.query_one("#status", Static).update(self._status_text())
        units.border_title = "units"
        units.update(self._units_text(max(units.content_size.height, 1)))
        self._paint_logs(logs)
        self.query_one("#footer", Static).update(self._footer_text())

    def _status_text(self) -> Text:
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(" bijou ", "bold reverse")
        mode_style = {Mode.NORMAL: "bold", Mode.COMMAND: "bold yellow", Mode.SEARCH: "bold cyan"}
        text.append(f" {self.key_mode} ", mode_style[self.key_mode])
        snap = self.snap
        if snap is None:
            text.append("  reading status...", "dim")
        else:
            for gpu in snap.gpus:
                share = gpu.used_mib / gpu.total_mib if gpu.total_mib else 0.0
                style = "red" if share > 0.8 else "yellow" if share > 0.5 else "green"
                text.append("  ● ", style)
                text.append(f"gpu {gpu.used_mib / 1024:.1f}/{gpu.total_mib / 1024:.1f}G")
                if gpu.holders:
                    counts: dict[str, int] = {}
                    for name in gpu.holders:
                        counts[name] = counts.get(name, 0) + 1
                    held = ", ".join(f"{c}x {n}" if c > 1 else n for n, c in counts.items())
                    text.append(f" ({held})", "dim")
            if not snap.gpus:
                text.append("  ○ no gpu", "dim")
            if not snap.checkpoint:
                text.append("  ● random weights", "yellow")
            elif snap.checkpoint_present:
                text.append("  ● ", "green")
                text.append(snap.checkpoint)
            else:
                text.append(f"  ○ {snap.checkpoint} missing", "red")
            text.append(f"  adapters {snap.adapters}/{snap.skills}")
            text.append(f"  full {snap.full_finetunes}/{snap.skills}")
            for name, alive in snap.services:
                text.append("  ● " if alive else "  ○ ", "green" if alive else "dim")
                text.append(name, "" if alive else "dim")
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
            name_style = "reverse" if i == self.selected else ""
            rows.append((Text.assemble((f" {glyph} ", style), (tracked.unit.id, name_style)), i))
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
        parts = [tracked.unit.id, str(tracked.status)]
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

    def _footer_text(self) -> Text:
        if self.key_mode is Mode.COMMAND:
            return Text(f":{self.typed}█")
        if self.key_mode is Mode.SEARCH:
            return Text(f"/{self.typed}█")
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
