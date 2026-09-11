"""The session index: SQLite, with an FTS5 table over requests, answers and step goals.

A record is stored whole as JSON, so reading one back is exact. The listing columns and the
full-text table are derived from it on every save.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

from engine.core.types.agent import RunStatus, SessionRecord, SessionSummary
from engine.core.types.errors import StoreError

SCHEMA = """
create table if not exists sessions (
    id text primary key,
    created_at text not null,
    updated_at text not null,
    status text not null,
    request text not null,
    answer text not null,
    record text not null
);
create index if not exists sessions_created on sessions (created_at);
create virtual table if not exists sessions_fts using fts5 (
    id unindexed, request, answer, goals, tokenize = 'porter unicode61'
);
"""


def fts_query(text: str) -> str:
    """Every word of text as a quoted term, so user input cannot be FTS syntax."""
    return " ".join(f'"{w}"' for w in re.findall(r"\w+", text.lower()))


class SqliteSessionStore:
    """Satisfies SessionStore."""

    def __init__(self, path: Path) -> None:
        try:
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise StoreError(f"session store at {path} could not be opened: {exc}") from exc
        self._lock = threading.Lock()

    def save(self, record: SessionRecord) -> None:
        goals = " ".join(s.goal for s in record.plan.steps) if record.plan else ""
        try:
            with self._lock, self._db:
                self._db.execute(
                    "insert or replace into sessions values (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.status.value,
                        record.request,
                        record.answer,
                        record.model_dump_json(),
                    ),
                )
                self._db.execute("delete from sessions_fts where id = ?", (record.id,))
                self._db.execute(
                    "insert into sessions_fts values (?, ?, ?, ?)",
                    (record.id, record.request, record.answer, goals),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"could not save session {record.id}: {exc}") from exc

    def get(self, session_id: str) -> SessionRecord | None:
        row = self._one("select record from sessions where id = ?", (session_id,))
        return SessionRecord.model_validate_json(row[0]) if row else None

    def recent(self, limit: int) -> list[SessionSummary]:
        rows = self._all(
            "select id, created_at, status, request, answer from sessions "
            "order by created_at desc limit ?",
            (limit,),
        )
        return [_summary(r) for r in rows]

    def search(self, query: str, limit: int) -> list[SessionSummary]:
        match = fts_query(query)
        if not match:
            return []
        rows = self._all(
            "select s.id, s.created_at, s.status, s.request, s.answer "
            "from sessions_fts f join sessions s on s.id = f.id "
            "where sessions_fts match ? order by f.rank limit ?",
            (match, limit),
        )
        return [_summary(r) for r in rows]

    def records(self, limit: int) -> list[SessionRecord]:
        rows = self._all("select record from sessions order by created_at desc limit ?", (limit,))
        return [SessionRecord.model_validate_json(r[0]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _one(self, sql: str, args: tuple[object, ...]) -> tuple[str, ...] | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def _all(self, sql: str, args: tuple[object, ...]) -> list[tuple[str, ...]]:
        try:
            with self._lock:
                return list(self._db.execute(sql, args).fetchall())
        except sqlite3.Error as exc:
            raise StoreError(f"session store query failed: {exc}") from exc


def _summary(row: tuple[str, ...]) -> SessionSummary:
    session_id, created_at, status, request, answer = row
    return SessionSummary.model_validate(
        {
            "id": session_id,
            "created_at": created_at,
            "status": RunStatus(status),
            "request": request,
            "answer": answer[:200],
        }
    )
