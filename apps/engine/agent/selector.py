"""The skill selector: which LoRA skills a step equips on the diffusion model.

The chat model reads the step and every trained skill's description and names at most
planning.max_skills of them, or a phase schedule over them, or none. Names are checked against
the skill server's catalog before anything is equipped. With no trained skill there is nothing to
choose, so no model call is made.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from engine.agent.llm import structured
from engine.agent.planner import skills_line
from engine.core.config import AgentConfig
from engine.core.protocols import ChatModel, TraceSink
from engine.core.types.agent import (
    PhaseSpec,
    PlanStep,
    RequestContext,
    SkillInfo,
    SkillPick,
    TraceEvent,
    TraceKind,
    Usage,
)
from engine.core.types.errors import ModelError, PlanError


class PickReply(BaseModel):
    """What the selector model returns."""

    skills: list[str] = Field(default_factory=list)
    schedule: list[PhaseSpec] | None = None
    reason: str = ""


def check_pick(reply: PickReply, trained: set[str], max_skills: int, schedules: bool) -> None:
    """Reject unknown or untrained names, too many skills, and malformed schedules."""
    if reply.skills and reply.schedule is not None:
        raise PlanError("give skills or a schedule, not both")
    named = list(reply.skills)
    if reply.schedule is not None:
        if not schedules:
            raise PlanError("schedules are not allowed; list skills instead")
        if not reply.schedule:
            raise PlanError("an empty schedule equips nothing; give skills: [] instead")
        for phase in reply.schedule:
            if not 0.0 <= phase.start < phase.end <= 1.0:
                raise PlanError(f"phase [{phase.start}, {phase.end}) is not inside [0, 1)")
            if not phase.skills or any(w <= 0 for w in phase.skills.values()):
                raise PlanError("every phase needs at least one skill with a positive weight")
            named += list(phase.skills)
    unknown = sorted({n for n in named if n not in trained})
    if unknown:
        raise PlanError(
            f"cannot equip {', '.join(unknown)}; choose from {', '.join(sorted(trained))}"
        )
    if len(set(named)) > max_skills:
        raise PlanError(f"{len(set(named))} skills; at most {max_skills}")


class SkillSelector:
    """Picks the skills one step equips."""

    def __init__(self, model: ChatModel, cfg: AgentConfig, trace: TraceSink) -> None:
        self.model = model
        self.cfg = cfg
        self.trace = trace

    async def pick(
        self, ctx: RequestContext, step: PlanStep, skills: list[SkillInfo]
    ) -> tuple[SkillPick, Usage]:
        """The pick for one step, or an empty pick with the reason none was made."""
        planning = self.cfg.planning
        trained = {s.name for s in skills if s.trained}
        usage = Usage()
        started = time.monotonic()
        if planning.max_skills == 0:
            pick = SkillPick(reason="equipping is off: agent.planning.max_skills is 0")
        elif not trained:
            pick = SkillPick(reason="the skill server has no trained skills")
        else:
            system = self.cfg.prompt.selector.format(
                max_skills=planning.max_skills,
                schedule_hint=self.cfg.prompt.schedule_hint if planning.allow_schedules else "",
                skills=skills_line(skills),
            )
            try:
                reply, usage = await structured(
                    self.model,
                    ctx,
                    schema=PickReply,
                    system=system,
                    user=f"Step: {step.goal}",
                    loop=self.cfg.loop,
                    trace=self.trace,
                    step_id=step.id,
                    purpose="select",
                    max_tokens=self.cfg.llm.max_tokens,
                    temperature=0.0,
                    check=lambda r: check_pick(
                        r, trained, planning.max_skills, planning.allow_schedules
                    ),
                )
                pick = SkillPick(skills=reply.skills, schedule=reply.schedule, reason=reply.reason)
            except (PlanError, ModelError) as exc:
                pick = SkillPick(reason=f"selector failed, nothing equipped: {exc}")
        self.trace.emit(
            TraceEvent(
                kind=TraceKind.SKILLS_PICKED,
                session_id=ctx.session_id,
                step_id=step.id,
                data={
                    "skills": pick.equipped,
                    "schedule": [p.model_dump() for p in pick.schedule or []],
                    "reason": pick.reason,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        )
        return pick, usage
