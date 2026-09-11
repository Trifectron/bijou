"""Starts, stops and streams units.

Each unit runs as the launcher followed by its args, just by default, in its own
session. A stop signals the whole process group, so just, uv, python and anything
they spawned all receive it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Callable, Sequence
from pathlib import Path

from bijou.experiments.console.logs import Stream

LineSink = Callable[[str, Stream, str], None]
ExitSink = Callable[[str, int | None], None]

_LINE_LIMIT = 1 << 20


class Runner:
    """Owns the processes the console started, by unit id."""

    def __init__(
        self,
        root: Path,
        on_line: LineSink,
        on_exit: ExitSink,
        launcher: Sequence[str] = ("just",),
    ) -> None:
        self.root = root
        self.on_line = on_line
        self.on_exit = on_exit
        self.launcher = tuple(launcher)
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def owns(self, unit_id: str) -> bool:
        """Whether a process started for this unit has not exited yet."""
        return unit_id in self._procs

    async def start(self, unit_id: str, args: Sequence[str]) -> None:
        """Spawn the unit and stream its output until it exits. Raises OSError on spawn."""
        cmd = [*self.launcher, *args]
        self.on_line(unit_id, "meta", "$ " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.root,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=os.environ | {"PYTHONUNBUFFERED": "1"},
            limit=_LINE_LIMIT,
        )
        self._procs[unit_id] = proc
        task = asyncio.create_task(self._watch(unit_id, proc))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def stop(self, unit_id: str) -> bool:
        """Send SIGTERM to the unit's process group. False when it is not running."""
        proc = self._procs.get(unit_id)
        if proc is None:
            return False
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        return True

    def shutdown(self) -> None:
        """Stop every unit still running."""
        for unit_id in list(self._procs):
            self.stop(unit_id)

    async def _pump(self, unit_id: str, reader: asyncio.StreamReader, stream: Stream) -> None:
        while True:
            try:
                raw = await reader.readline()
            except ValueError:
                raw = await reader.read(_LINE_LIMIT)
            if not raw:
                return
            text = raw.decode(errors="replace").rstrip("\r\n")
            # A progress bar redraws with carriage returns; the last frame is the current one.
            self.on_line(unit_id, stream, text.rsplit("\r", 1)[-1])

    async def _watch(self, unit_id: str, proc: asyncio.subprocess.Process) -> None:
        pumps = []
        if proc.stdout is not None:
            pumps.append(self._pump(unit_id, proc.stdout, "out"))
        if proc.stderr is not None:
            pumps.append(self._pump(unit_id, proc.stderr, "err"))
        await asyncio.gather(*pumps)
        code = await proc.wait()
        self._procs.pop(unit_id, None)
        self.on_exit(unit_id, code)
