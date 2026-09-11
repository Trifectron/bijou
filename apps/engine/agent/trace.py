"""Trace sinks: one JSONL file per session, an in-memory collector per run, and a fan-out.

A trace sink must not stop a run. A file that cannot be written drops the event.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from engine.core.protocols import TraceSink
from engine.core.types.agent import TraceEvent


class JsonlTrace:
    """Appends each event to dir/<session_id>.jsonl. Satisfies TraceSink."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def emit(self, event: TraceEvent) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{event.session_id}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")
        except OSError:
            return


class Collector:
    """Holds events by session until the run that produced them takes them. Satisfies TraceSink."""

    def __init__(self) -> None:
        self._events: defaultdict[str, list[TraceEvent]] = defaultdict(list)

    def emit(self, event: TraceEvent) -> None:
        self._events[event.session_id].append(event)

    def take(self, session_id: str) -> list[TraceEvent]:
        """Every event held for a session, removed."""
        return self._events.pop(session_id, [])


class Fanout:
    """Sends each event to every sink. Satisfies TraceSink."""

    def __init__(self, sinks: Sequence[TraceSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, event: TraceEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)
