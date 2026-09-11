"""The interfaces every implementation satisfies.

The diffusion model's seams:

  AdapterSite       how a delta attaches to one frozen base module
  ActivationPolicy  what is live at denoising step t
  Grader            how a skill's output is scored
  Skill             data, a grader and a prompt template, built in or collected
  Backend           a base model, so nanoDiff's internals are touched in one package

The agent's seams, each with a double in core.doubles:

  ChatModel         the LLM that plans, picks skills, and drives each subagent's loop
  SkillRuntime      the diffusion model with its bank of skills, in process or over HTTP
  Tool              one capability a subagent may call
  Policy            what may run, what is denied, and what waits for the user
  TraceSink         where every event goes
  SessionStore      the index of every session
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from engine.core.types.agent import (
    Decision,
    ModelRequest,
    ModelResponse,
    ProposedAction,
    RequestContext,
    SessionRecord,
    SessionSummary,
    SkillInfo,
    SkillRequest,
    SkillResult,
    ToolDefinition,
    ToolOutput,
    TraceEvent,
)
from engine.core.types.diffusion import AdapterSpec, GenerationRequest, Sample, Score, Split

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------- the diffusion model


@runtime_checkable
class AdapterSite(Protocol):
    """A frozen module with a name-keyed bank of deltas attached to it.

    LoRA is one implementation. Another must not require changes outside engine.adapters.
    """

    def add(self, spec: AdapterSpec) -> None:
        """Attach a new delta. It must be a no-op until trained."""
        ...

    def names(self) -> Sequence[str]:
        """Every delta attached here."""
        ...

    def parameters_for(self, name: str) -> Iterator[torch.nn.Parameter]:
        """The trainable parameters of one delta."""
        ...


@runtime_checkable
class ActivationPolicy(Protocol):
    """Decides the active adapter set at a point along the denoising trajectory.

    Static application is the degenerate case: a policy that ignores progress.
    """

    def active_at(self, progress: float) -> dict[str, float]:
        """Adapter name to weight, for progress in [0, 1). 0 is fully masked."""
        ...


@runtime_checkable
class Grader(Protocol):
    """Scores one generation against one sample. No model, no GPU, no network."""

    def __call__(self, sample: Sample, output: str) -> Score: ...


@runtime_checkable
class SkillData(Protocol):
    """Generates a skill's train, dev and eval splits from a seed."""

    def generate(self, n: int, seed: int, split: Split = "train") -> list[Sample]: ...


class Skill(Protocol):
    """Everything a skill exposes: data, a grader, and the text around a prompt.

    A skill module satisfies it, and so does a dataset skill read from disk. DESCRIPTION is what
    the agent's skill selector reads when it decides whether to equip the skill.
    """

    @property
    def NAME(self) -> str: ...  # noqa: N802 - module constant

    @property
    def DESCRIPTION(self) -> str: ...  # noqa: N802 - module constant

    @property
    def INSTRUCTIONS(self) -> tuple[str, ...]: ...  # noqa: N802 - module constant

    @property
    def PROMPT(self) -> str: ...  # noqa: N802 - module constant

    def generate(self, n: int, seed: int, split: Split = "train") -> list[Sample]: ...

    def grade(self, sample: Sample, output: str) -> Score: ...


@runtime_checkable
class Backend(Protocol):
    """A base model, its training step, and its sampler."""

    def build(self) -> torch.nn.Module:
        """Construct the base model on the configured device."""
        ...

    def generate(self, req: GenerationRequest, on_step: object | None = None) -> str:
        """Run the reverse process. on_step is called before each denoising step."""
        ...


# ---------------------------------------------------------------- the agent


@runtime_checkable
class ChatModel(Protocol):
    """One chat completion. Raises ModelError; retryable ones are retried by the caller."""

    async def generate(self, ctx: RequestContext, req: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class SkillRuntime(Protocol):
    """The skill bank. Raises SkillRuntimeError."""

    async def catalog(self) -> list[SkillInfo]: ...

    async def run(self, ctx: RequestContext, req: SkillRequest) -> SkillResult: ...


@runtime_checkable
class Tool(Protocol):
    """One capability. Raises ToolError, which the loop hands back to the model."""

    @property
    def definition(self) -> ToolDefinition: ...

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput: ...


@runtime_checkable
class Policy(Protocol):
    """Allows, denies, or holds an action for confirmation."""

    def authorize(self, action: ProposedAction) -> Decision: ...


@runtime_checkable
class TraceSink(Protocol):
    """Receives every event. Must not raise."""

    def emit(self, event: TraceEvent) -> None: ...


@runtime_checkable
class SessionStore(Protocol):
    """Sessions by id, newest first, by full-text search, and in full for mining."""

    def save(self, record: SessionRecord) -> None: ...

    def get(self, session_id: str) -> SessionRecord | None: ...

    def recent(self, limit: int) -> list[SessionSummary]: ...

    def search(self, query: str, limit: int) -> list[SessionSummary]: ...

    def records(self, limit: int) -> list[SessionRecord]: ...
