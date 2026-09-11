"""Per-unit log buffers with search, and the files every line is mirrored to."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Literal, TextIO

Stream = Literal["out", "err", "meta"]


@dataclass(frozen=True)
class LogLine:
    """One line of unit output. meta lines are the console's own notes."""

    at: datetime
    stream: Stream
    text: str


class LogBuffer:
    """The newest lines of one unit, bounded."""

    def __init__(self, limit: int) -> None:
        self._lines: deque[LogLine] = deque(maxlen=limit)

    def __len__(self) -> int:
        return len(self._lines)

    def append(self, line: LogLine) -> None:
        self._lines.append(line)

    def clear(self) -> None:
        self._lines.clear()

    def window(self, start: int, rows: int) -> list[LogLine]:
        """Up to rows lines from index start."""
        return list(islice(self._lines, start, start + rows))

    def find(self, query: str, start: int, backwards: bool = False) -> int | None:
        """The first line from start containing query, ignoring case, wrapping around."""
        count = len(self._lines)
        if not query or not count:
            return None
        needle = query.lower()
        step = -1 if backwards else 1
        for k in range(count):
            i = (start + step * k) % count
            if needle in self._lines[i].text.lower():
                return i
        return None


def log_name(unit_id: str) -> str:
    """The file name for one unit's log."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", unit_id).strip("_") + ".log"


class LogWriter:
    """Appends every line to one file per unit under a directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._files: dict[str, TextIO] = {}

    def append(self, unit_id: str, line: LogLine) -> None:
        handle = self._files.get(unit_id)
        if handle is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Held open for the console's lifetime and closed by close.
            handle = (self.directory / log_name(unit_id)).open(  # noqa: SIM115 - closed in close
                "a", encoding="utf-8", buffering=1
            )
            self._files[unit_id] = handle
        handle.write(f"{line.at:%Y-%m-%dT%H:%M:%S} {line.stream} {line.text}\n")

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
