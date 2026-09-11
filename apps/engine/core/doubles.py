"""Test doubles for every protocol in core.protocols. No network, no model, no disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from engine.core.types.agent import (
    ModelRequest,
    ModelResponse,
    RequestContext,
    RiskClass,
    SessionRecord,
    SessionSummary,
    SkillInfo,
    SkillRequest,
    SkillResult,
    ToolCall,
    ToolDefinition,
    ToolOutput,
    TraceEvent,
    TraceKind,
    Usage,
)
from engine.core.types.errors import ModelError

Reply = ModelResponse | ModelError | Callable[[ModelRequest], ModelResponse]


def text(content: str) -> ModelResponse:
    """A reply with no tool calls."""
    return ModelResponse(content=content, usage=Usage(prompt_tokens=10, completion_tokens=5))


def calls(*tool_calls: tuple[str, dict[str, Any]], content: str = "") -> ModelResponse:
    """A reply asking for tool calls, each a name and its arguments."""
    return ModelResponse(
        content=content,
        tool_calls=[
            ToolCall(id=f"call-{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(tool_calls)
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


class ScriptedModel:
    """Replies in order. A ModelError in the script is raised. Satisfies ChatModel."""

    def __init__(self, replies: Sequence[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []

    async def generate(self, ctx: RequestContext, req: ModelRequest) -> ModelResponse:  # noqa: ARG002 - protocol signature
        self.requests.append(req)
        if not self.replies:
            raise AssertionError("the scripted model ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, ModelError):
            raise reply
        if isinstance(reply, ModelResponse):
            return reply
        return reply(req)


class RoutedModel:
    """Replies by which prompt it is asked: the first system line picks the handler.

    For tests that run whole requests, where planner, selector and subagents interleave.
    """

    def __init__(self, routes: dict[str, Callable[[ModelRequest], ModelResponse]]) -> None:
        self.routes = routes
        self.requests: list[ModelRequest] = []

    async def generate(self, ctx: RequestContext, req: ModelRequest) -> ModelResponse:  # noqa: ARG002 - protocol signature
        self.requests.append(req)
        system = req.messages[0].content if req.messages else ""
        for marker, handler in self.routes.items():
            if marker in system:
                return handler(req)
        raise AssertionError(f"no route for a prompt starting {system[:60]!r}")


class FakeSkillRuntime:
    """A skill server with fixed skills. Satisfies SkillRuntime."""

    def __init__(
        self,
        skills: Sequence[SkillInfo] = (),
        reply: Callable[[SkillRequest], str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.skills = list(skills)
        self.reply = reply or (lambda req: f"[{'+'.join(req.skills)}] {req.prompt}")
        self.error = error
        self.requests: list[SkillRequest] = []

    async def catalog(self) -> list[SkillInfo]:
        if self.error is not None:
            raise self.error
        return list(self.skills)

    async def run(self, ctx: RequestContext, req: SkillRequest) -> SkillResult:  # noqa: ARG002 - protocol signature
        if self.error is not None:
            raise self.error
        self.requests.append(req)
        live = req.skills or [n for p in req.schedule or [] for n in p.skills]
        return SkillResult(text=self.reply(req), skills=sorted(set(live)))


class FakeTool:
    """A tool that records its calls and answers from a function. Satisfies Tool."""

    def __init__(
        self,
        name: str,
        risk: RiskClass = RiskClass.READ_PUBLIC,
        answer: Callable[[dict[str, Any]], str] | None = None,
        sequential: bool = False,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description=f"the {name} tool",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            risk=risk,
            sequential=sequential,
        )
        self.answer = answer or (lambda args: f"{name} says {args}")
        self.error = error
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:  # noqa: ARG002 - protocol signature
        import asyncio

        self.calls.append(arguments)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return ToolOutput(text=self.answer(arguments))


class MemoryTrace:
    """Keeps every event in a list. Satisfies TraceSink."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[TraceKind]:
        return [e.kind for e in self.events]


class MemorySessionStore:
    """Sessions in a dict. Search is a substring match. Satisfies SessionStore."""

    def __init__(self) -> None:
        self.saved: dict[str, SessionRecord] = {}

    def save(self, record: SessionRecord) -> None:
        self.saved[record.id] = record.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        found = self.saved.get(session_id)
        return found.model_copy(deep=True) if found else None

    def _newest(self) -> list[SessionRecord]:
        return sorted(self.saved.values(), key=lambda r: r.created_at, reverse=True)

    def recent(self, limit: int) -> list[SessionSummary]:
        return [_summary(r) for r in self._newest()[:limit]]

    def search(self, query: str, limit: int) -> list[SessionSummary]:
        needle = query.lower()
        found = [r for r in self._newest() if needle in (r.request + " " + r.answer).lower()]
        return [_summary(r) for r in found[:limit]]

    def records(self, limit: int) -> list[SessionRecord]:
        return self._newest()[:limit]


def _summary(record: SessionRecord) -> SessionSummary:
    return SessionSummary(
        id=record.id,
        created_at=record.created_at,
        status=record.status,
        request=record.request,
        answer=record.answer[:200],
    )
