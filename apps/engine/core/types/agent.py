"""Data that crosses a module boundary as a value: messages, plans, picks, results, events.

Every type here is plain data. Objects that own state live beside their implementation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- risk


class RiskClass(StrEnum):
    """What running a tool can do, in increasing order of consequence."""

    READ_PUBLIC = "read_public"
    READ_AUTHENTICATED = "read_authenticated"
    PREPARE_WRITE = "prepare_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"
    FORBIDDEN = "forbidden"

    @property
    def rank(self) -> int:
        return list(RiskClass).index(self)

    def at_least(self, other: RiskClass) -> bool:
        return self.rank >= other.rank


# ---------------------------------------------------------------- messages and model calls


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """One tool call the model asked for. error holds why its arguments did not parse."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class Message(BaseModel):
    """One turn of a conversation with the chat model."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=call.id, name=call.name)


class ToolDefinition(BaseModel):
    """What the model is told about a tool, and what the policy is told about its risk."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: RiskClass = RiskClass.READ_PUBLIC
    # A sequential tool holds state between calls, so a step calling it runs its calls in order.
    sequential: bool = False
    source: str = "builtin"


class ToolOutput(BaseModel):
    """What a tool returns to the model."""

    text: str
    data: dict[str, Any] | None = None


class Usage(BaseModel):
    """Tokens spent."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def plus(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


class ModelRequest(BaseModel):
    """One chat completion. json_schema asks for structured output and excludes tools."""

    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.2
    json_schema: dict[str, Any] | None = None


class ModelResponse(BaseModel):
    """One chat completion's reply."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    model: str = ""

    def as_message(self) -> Message:
        return Message.assistant(self.content, self.tool_calls)


# ---------------------------------------------------------------- skills


class SkillInfo(BaseModel):
    """One skill the skill server knows. Only a trained skill can be equipped."""

    name: str
    description: str
    trained: bool


class PhaseSpec(BaseModel):
    """Skills live over the fractional interval [start, end) of the denoising trajectory."""

    start: float
    end: float
    skills: dict[str, float]


class SkillRequest(BaseModel):
    """One generation on the diffusion model with skills equipped."""

    prompt: str
    skills: list[str] = Field(default_factory=list)
    schedule: list[PhaseSpec] | None = None
    instruct: bool = True
    gen_length: int | None = None
    steps: int | None = None


class SkillResult(BaseModel):
    """What the diffusion model generated, and which skills were live."""

    text: str
    skills: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    gen_length: int = 0
    steps: int = 0


class SkillPick(BaseModel):
    """What the selector equipped for one step: skills, or a schedule over them, or nothing."""

    skills: list[str] = Field(default_factory=list)
    schedule: list[PhaseSpec] | None = None
    reason: str = ""

    @property
    def equipped(self) -> list[str]:
        """Every skill live at any point, in the order first named."""
        if self.schedule is None:
            return list(self.skills)
        seen: list[str] = []
        for phase in self.schedule:
            seen += [n for n in phase.skills if n not in seen]
        return seen


# ---------------------------------------------------------------- plans and results


class PlanStep(BaseModel):
    """One unit of work, run by one subagent. depends_on names earlier steps."""

    id: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """The steps a request was split into. fallback is set when planning failed."""

    steps: list[PlanStep]
    reason: str = ""
    fallback: bool = False


class RunStatus(StrEnum):
    """How a step or a run ended."""

    RUNNING = "running"
    ANSWERED = "answered"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    STEP_LIMIT = "step_limit"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    STALLED = "stalled"
    ERROR = "error"


class ProposedAction(BaseModel):
    """A tool call as the policy sees it."""

    tool: str
    arguments: dict[str, Any]
    risk: RiskClass


class ConfirmationRequest(BaseModel):
    """An action held for the user. Bound to one exact payload, single use, short lived."""

    token: str
    tool: str
    arguments: dict[str, Any]
    payload_hash: str
    risk: RiskClass
    summary: str
    expires_at: datetime


class Decision(BaseModel):
    """What the policy decided for one action."""

    kind: Literal["allow", "deny", "confirm"]
    reason: str = ""
    confirmation: ConfirmationRequest | None = None

    @classmethod
    def allow(cls) -> Decision:
        return cls(kind="allow")

    @classmethod
    def deny(cls, reason: str) -> Decision:
        return cls(kind="deny", reason=reason)

    @classmethod
    def confirm(cls, request: ConfirmationRequest) -> Decision:
        return cls(kind="confirm", confirmation=request)


class StepReport(BaseModel):
    """How one step went: what it equipped, what it called, and what it answered."""

    id: str
    goal: str
    status: RunStatus
    answer: str = ""
    skills: list[str] = Field(default_factory=list)
    schedule: list[PhaseSpec] | None = None
    pick_reason: str = ""
    tools: list[str] = Field(default_factory=list)
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)


class PendingStep(BaseModel):
    """A step stopped on an action awaiting confirmation, with everything needed to resume it."""

    step_id: str
    goal: str
    messages: list[Message]
    pick: SkillPick
    call: ToolCall
    confirmation: ConfirmationRequest
    turns: int
    tools: list[str]
    usage: Usage


# ---------------------------------------------------------------- traces


class TraceKind(StrEnum):
    RUN_STARTED = "run_started"
    PLANNED = "planned"
    SKILLS_PICKED = "skills_picked"
    MODEL_CALL = "model_call"
    MODEL_ERROR = "model_error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    POLICY = "policy"
    STEP_DONE = "step_done"
    CONFIRMED = "confirmed"
    RUN_DONE = "run_done"
    NOTICE = "notice"


def now() -> datetime:
    return datetime.now(UTC)


class TraceEvent(BaseModel):
    """One thing that happened in a run. One JSONL line per event."""

    kind: TraceKind
    session_id: str
    step_id: str = ""
    at: datetime = Field(default_factory=now)
    data: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """What one run or confirmation returns. The harness HTTP surface sends this as JSON."""

    session_id: str
    request: str
    status: RunStatus
    answer: str
    plan: Plan
    steps: list[StepReport]
    usage: Usage
    duration_ms: int
    confirmation: ConfirmationRequest | None = None
    events: list[TraceEvent] = Field(default_factory=list)


# ---------------------------------------------------------------- sessions and patterns


class SessionRecord(BaseModel):
    """Everything about one session, as stored."""

    id: str
    created_at: datetime
    updated_at: datetime
    request: str
    status: RunStatus
    answer: str = ""
    resumed_from: str | None = None
    plan: Plan | None = None
    steps: list[StepReport] = Field(default_factory=list)
    # Steps stopped on an action awaiting the user, confirmed one at a time.
    pending: list[PendingStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class SessionSummary(BaseModel):
    """One line of a session listing."""

    id: str
    created_at: datetime
    status: RunStatus
    request: str
    answer: str


class SkillPair(BaseModel):
    """One worked example in a skill proposal."""

    input: str
    output: str


class SkillProposal(BaseModel):
    """A skill the pattern miner proposes. The file format bijou collect reads.

    approved is always written false. A person sets it after reading the proposal.
    """

    name: str
    description: str
    instruction: str
    examples: list[str] = Field(default_factory=list)
    pairs: list[SkillPair] = Field(default_factory=list)
    approved: bool = False
    source: str = "harness patterns"
    occurrences: int = 0
    sessions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- the request context


@dataclass
class RequestContext:
    """Per-run state every component receives. Never shared between runs."""

    session_id: str
    deadline: float
    user_id: str = "local"
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def start(cls, session_id: str, timeout_secs: float, user_id: str = "local") -> RequestContext:
        return cls(session_id=session_id, deadline=time.monotonic() + timeout_secs, user_id=user_id)

    def remaining(self) -> float:
        """Seconds until the deadline, never negative."""
        return max(self.deadline - time.monotonic(), 0.0)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()
