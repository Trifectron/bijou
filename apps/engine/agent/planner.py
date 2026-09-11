"""The planner: how many subagents a request needs, and what each one does.

A plan is a list of steps whose dependencies point only at earlier steps, so it is acyclic by
construction. A plan that cannot be produced becomes one step holding the whole request, with
fallback set and the reason recorded, so a planning failure is visible and never fatal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engine.agent.task import structured
from engine.core.config import AgentConfig
from engine.core.protocols import ChatModel, TraceSink
from engine.core.types.agent import (
    Plan,
    PlanStep,
    RequestContext,
    SkillInfo,
    ToolDefinition,
    TraceEvent,
    TraceKind,
    Usage,
)
from engine.core.types.errors import ModelError, PlanError


class PlanReply(BaseModel):
    """What the planner model returns."""

    steps: list[PlanStep] = Field(min_length=1)
    reason: str = ""


def check_plan(reply: PlanReply, max_steps: int) -> None:
    """Reject a plan that is too long, repeats an id, or depends on a later step."""
    if len(reply.steps) > max_steps:
        raise PlanError(f"{len(reply.steps)} steps; at most {max_steps}")
    seen: set[str] = set()
    for step in reply.steps:
        if not step.id.strip() or not step.goal.strip():
            raise PlanError("every step needs an id and a goal")
        if step.id in seen:
            raise PlanError(f"step id {step.id} is used twice")
        for dep in step.depends_on:
            if dep not in seen:
                raise PlanError(f"step {step.id} depends on {dep}, which is not an earlier step")
        seen.add(step.id)


def skills_line(skills: list[SkillInfo]) -> str:
    """The trained skills, one per line, as the planner and selector read them."""
    trained = [s for s in skills if s.trained]
    return "\n".join(f"- {s.name}: {s.description}" for s in trained) or "none"


class Planner:
    """Splits a request into steps with the chat model."""

    def __init__(self, model: ChatModel, cfg: AgentConfig, trace: TraceSink) -> None:
        self.model = model
        self.cfg = cfg
        self.trace = trace

    async def plan(
        self,
        ctx: RequestContext,
        request: str,
        context: str,
        tools: list[ToolDefinition],
        skills: list[SkillInfo],
    ) -> tuple[Plan, Usage]:
        """The plan, or a one-step fallback when the model cannot produce one."""
        max_steps = self.cfg.planning.max_steps
        system = self.cfg.prompt.planner.format(
            max_steps=max_steps,
            tools=", ".join(t.name for t in tools) or "none",
            skills="\n" + skills_line(skills),
        )
        user = request if not context else f"{request}\n\nFrom an earlier session:\n{context}"
        usage = Usage()
        try:
            reply, usage = await structured(
                self.model,
                ctx,
                schema=PlanReply,
                system=system,
                user=user,
                loop=self.cfg.loop,
                trace=self.trace,
                step_id="",
                purpose="plan",
                max_tokens=self.cfg.llm.max_tokens,
                temperature=self.cfg.llm.temperature,
                check=lambda r: check_plan(r, max_steps),
            )
            plan = Plan(steps=reply.steps, reason=reply.reason)
        except (PlanError, ModelError) as exc:
            plan = Plan(steps=[PlanStep(id="1", goal=request)], reason=str(exc), fallback=True)
        self.trace.emit(
            TraceEvent(
                kind=TraceKind.PLANNED,
                session_id=ctx.session_id,
                data={
                    "steps": [s.model_dump() for s in plan.steps],
                    "fallback": plan.fallback,
                    "reason": plan.reason,
                },
            )
        )
        return plan, usage
