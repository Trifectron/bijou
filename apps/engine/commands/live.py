"""The live trace: what the agent is doing, printed line by line while a request runs.

engine run and engine chat print it, so the plan the planner produced, the skills each step
equipped and how long that took, and every tool call are visible before the answer arrives. It
renders the same trace events that go to the trace files, to Prometheus and to Phoenix, so
anything shown here is in a span too.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from engine.core.types.agent import TraceEvent, TraceKind

STYLES: dict[str, str] = {
    "plan": "magenta",
    "equip": "cyan",
    "llm": "blue",
    "policy": "yellow",
    "tool": "yellow",
    "skill": "cyan",
    "step": "green",
    "confirm": "yellow",
    "notice": "red",
}
WIDTH = 7
SEP = " · "
ARGUMENT_CHARS = 80
REASON_CHARS = 120


def _clip(text: str, limit: int) -> str:
    """One line of at most limit characters, with an ellipsis when it was longer."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _secs(data: dict[str, object], key: str = "duration_ms") -> str:
    """A duration in seconds, or nothing when the event carries none."""
    ms = data.get(key)
    if not isinstance(ms, (int, float)) or isinstance(ms, bool):
        return ""
    return f"{float(ms) / 1000:.1f}s"


def _names(data: dict[str, object], key: str) -> list[str]:
    raw = data.get(key)
    return [str(n) for n in raw] if isinstance(raw, list) else []


def _joined(parts: list[str]) -> str:
    return SEP.join(p for p in parts if p)


def _plan(data: dict[str, object]) -> tuple[str, list[str]]:
    steps = [s for s in data.get("steps", []) if isinstance(s, dict)]
    plural = "" if len(steps) == 1 else "s"
    reason = _clip(str(data.get("reason", "")), REASON_CHARS)
    head = _joined(
        [
            f"{len(steps)} step{plural}",
            "planning fell back" if data.get("fallback") else "",
            reason,
        ]
    )
    lines = []
    for step in steps:
        after = [str(d) for d in step.get("depends_on") or []]
        lines.append(
            _joined([f"{step.get('id', '?')}  {step.get('goal', '')}", f"after {', '.join(after)}"])
            if after
            else f"{step.get('id', '?')}  {step.get('goal', '')}"
        )
    return head, lines


def _equipped(data: dict[str, object]) -> tuple[str, list[str]]:
    skills = _names(data, "skills")
    phases = [p for p in data.get("schedule", []) if isinstance(p, dict)]
    # A pick made without a model call, such as a step with no trained skill to choose from,
    # takes no measurable time and says so in its reason instead.
    chosen = f"chosen in {_secs(data)}" if data.get("duration_ms") else ""
    head = _joined(
        [
            "+".join(skills) or "nothing equipped",
            chosen,
            _clip(str(data.get("reason", "")), REASON_CHARS),
        ]
    )
    lines = [
        f"{float(p.get('start', 0)):.2f}-{float(p.get('end', 0)):.2f}  "
        + ", ".join(f"{n} x{w}" for n, w in (p.get("skills") or {}).items())
        for p in phases
    ]
    return head, lines


def _model_call(data: dict[str, object]) -> str:
    calls = _names(data, "tool_calls")
    return _joined(
        [
            str(data.get("purpose", "")),
            _secs(data),
            f"{data.get('prompt_tokens', 0)}+{data.get('completion_tokens', 0)} tokens",
            f"asked for {', '.join(calls)}" if calls else "",
        ]
    )


def _tool_result(data: dict[str, object]) -> str:
    """A finished call. run_skill reports what it equipped and how long equipping cost."""
    parts = [str(data.get("tool", "")), "ok" if data.get("ok") else "failed"]
    generate = _secs(data, "generate_ms")
    if generate:
        total = data.get("duration_ms", 0)
        spent = data.get("generate_ms", 0)
        equipping = (float(total) - float(spent)) / 1000 if isinstance(total, (int, float)) else 0.0
        parts += [
            "+".join(_names(data, "skills")) or "nothing",
            f"equipped in {max(equipping, 0.0):.1f}s",
            f"generated in {generate}",
        ]
    else:
        parts += [_secs(data), f"{data.get('chars', 0)} chars"]
    return _joined(parts)


def render(event: TraceEvent) -> tuple[str, str, list[str]] | None:
    """The label, the headline and any lines under it, or None for an event with nothing to show."""
    data = event.data
    if event.kind is TraceKind.PLANNED:
        head, lines = _plan(data)
        return "plan", head, lines
    if event.kind is TraceKind.SKILLS_PICKED:
        head, lines = _equipped(data)
        return "equip", head, lines
    if event.kind is TraceKind.MODEL_CALL:
        return "llm", _model_call(data), []
    if event.kind is TraceKind.MODEL_ERROR:
        retried = "retrying" if data.get("retried") else "giving up"
        return (
            "llm",
            _joined([str(data.get("purpose", "")), str(data.get("error", "")), retried]),
            [],
        )
    if event.kind is TraceKind.POLICY:
        return (
            "policy",
            _joined([str(data.get("tool", "")), str(data.get("decision", ""))]),
            [],
        )
    if event.kind is TraceKind.TOOL_CALL:
        arguments = _clip(str(data.get("arguments", "")), ARGUMENT_CHARS)
        return "tool", _joined([f"calling {data.get('tool', '')}", arguments]), []
    if event.kind is TraceKind.TOOL_RESULT:
        label = "skill" if data.get("generate_ms") is not None else "tool"
        return label, _tool_result(data), []
    if event.kind is TraceKind.STEP_DONE:
        head = _joined(
            [
                str(data.get("status", "")),
                f"{data.get('turns', 0)} turns",
                "+".join(_names(data, "skills")),
                ", ".join(dict.fromkeys(_names(data, "tools"))),
            ]
        )
        return "step", head, []
    if event.kind is TraceKind.CONFIRMED:
        approved = "approved" if data.get("approved") else "declined"
        return "confirm", _joined([str(data.get("tool", "")), approved]), []
    if event.kind is TraceKind.NOTICE:
        return "notice", str(data.get("message", "")), []
    return None


class LiveTrace:
    """Prints each event as it happens. Satisfies TraceSink."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def emit(self, event: TraceEvent) -> None:
        rendered = render(event)
        if rendered is None:
            return
        label, head, lines = rendered
        text = Text()
        text.append(label.ljust(WIDTH), STYLES.get(label, "dim"))
        if event.step_id:
            text.append(f"[{event.step_id}] ", "dim")
        text.append(head)
        for line in lines:
            text.append("\n" + " " * WIDTH + line, "dim")
        self.console.print(text)
