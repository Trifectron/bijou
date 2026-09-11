"""The chat pane: what the agent process says, as transcript turns and one-line log entries.

The agent unit runs engine chat --jsonl. Each line it prints is one JSON message: ready, event,
result or error. Events go to the agent's log; results and errors become turns. The console
writes ask, confirm and new messages back on its stdin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from rich.text import Text

Role = Literal["you", "agent", "note"]

LABELS: dict[Role, tuple[str, str]] = {
    "you": ("you", "bold green"),
    "agent": ("agent", "bold cyan"),
    "note": ("console", "bold yellow"),
}

# Event fields too long or too nested for one log line; Phoenix and the trace files keep them.
_LONG = frozenset({"input", "output", "answer", "arguments", "result", "messages", "request"})


@dataclass(frozen=True)
class Turn:
    """One entry in the transcript. meta is the dim line under an answer."""

    role: Role
    text: str
    meta: str = ""


def decode(line: str) -> dict[str, Any] | None:
    """One protocol message, or None for a line that is not one."""
    if not line.startswith("{"):
        return None
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return None
    return message if isinstance(message, dict) and "type" in message else None


def describe(event: dict[str, Any]) -> str:
    """A trace event as one log line: its kind, its step, and its short fields."""
    parts = [str(event.get("kind", "event"))]
    if event.get("step_id"):
        parts.append(f"[{event['step_id']}]")
    data = event.get("data")
    for key, value in (data if isinstance(data, dict) else {}).items():
        if key in _LONG:
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            value = "+".join(value) or "none"
        if isinstance(value, str | int | float | bool) and len(str(value)) <= 80:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summary(result: dict[str, Any]) -> str:
    """The dim line under an answer: status, steps, tokens, time."""
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    tokens = int(_number(usage.get("prompt_tokens")) + _number(usage.get("completion_tokens")))
    steps = len(result.get("steps") or [])
    seconds = _number(result.get("duration_ms")) / 1000
    plural = "" if steps == 1 else "s"
    return f"{result.get('status', '')} · {steps} step{plural} · {tokens} tokens · {seconds:.1f}s"


@dataclass
class Transcript:
    """The conversation as the chat pane shows it, and what the agent is doing."""

    turns: list[Turn] = field(default_factory=list)
    ready: bool = False
    waiting: bool = False
    activity: str = ""
    pending: str | None = None
    session: str | None = None

    def ask(self, text: str) -> None:
        self.turns.append(Turn("you", text))
        self.waiting, self.activity = True, ""

    def confirming(self, approve: bool) -> None:
        self.turns.append(Turn("you", "approved" if approve else "denied"))
        self.pending = None
        self.waiting, self.activity = True, ""

    def new(self) -> None:
        self.turns.clear()
        self.session = self.pending = None

    def note(self, text: str) -> None:
        self.turns.append(Turn("note", text))

    def stopped(self, code: int | None) -> None:
        self.ready = self.waiting = False
        self.pending = None
        self.note(f"the agent stopped (exit {code}); select it and press enter to start it again")

    def restarted(self) -> None:
        """A fresh agent process answers from a fresh session, so the thread ends here."""
        had_session = self.session is not None
        self.ready = self.waiting = False
        self.session = self.pending = None
        if had_session:
            self.note("the agent restarted; the next message starts a new conversation")

    def apply(self, message: dict[str, Any]) -> str | None:
        """Take one message from the agent. Returns the line for the agent's log, if any."""
        kind = message.get("type")
        if kind == "ready":
            self.ready = True
            metrics = message.get("metrics")
            return "ready" + (f", metrics at {metrics}" if metrics else "")
        if kind == "event":
            event = message.get("event")
            event = event if isinstance(event, dict) else {}
            self.activity = str(event.get("kind", "")).replace("_", " ")
            return describe(event)
        if kind == "result":
            result = message.get("result")
            result = result if isinstance(result, dict) else {}
            self.waiting, self.activity = False, ""
            self.session = str(result.get("session_id") or "") or self.session
            confirmation = result.get("confirmation")
            self.pending = (
                f"{confirmation.get('tool')}: {confirmation.get('summary')}"
                if isinstance(confirmation, dict)
                else None
            )
            answer = str(result.get("answer") or "")
            if not answer:
                answer = "waiting for your approval" if self.pending else "no answer"
            self.turns.append(Turn("agent", answer, summary(result)))
            return f"result {result.get('status')} · session {self.session}"
        if kind == "error":
            self.waiting, self.activity = False, ""
            text = str(message.get("message", ""))
            self.note(text)
            return f"error {text}"
        return None

    def render(self, spinner: str) -> Text:
        """Every turn, then what the agent is doing or waiting on."""
        out = Text()
        for turn in self.turns:
            if out.plain:
                out.append("\n\n")
            label, style = LABELS[turn.role]
            out.append(f"{label}  ", style)
            out.append(turn.text)
            if turn.meta:
                out.append(f"\n{turn.meta}", "dim")
        if self.waiting:
            out.append("\n\n" if out.plain else "")
            out.append(f"agent {spinner} ", "bold cyan")
            out.append(self.activity or "thinking", "dim")
        if self.pending:
            out.append("\n\n" if out.plain else "")
            out.append("needs approval  ", "bold yellow")
            out.append(self.pending)
            out.append("\na approve · d deny", "dim")
        return out
