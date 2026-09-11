"""run_skill: the tool through which a subagent uses the skills it equipped.

It exists only for a step whose pick equipped something, and sends its input to the diffusion
model with exactly that pick live: the listed skills, or the phase schedule over them.
"""

from __future__ import annotations

from typing import Any

from engine.core.protocols import SkillRuntime
from engine.core.types.agent import (
    RequestContext,
    RiskClass,
    SkillPick,
    SkillRequest,
    ToolDefinition,
    ToolOutput,
)
from engine.core.types.errors import SkillRuntimeError, ToolError

NAME = "run_skill"


class SkillTool:
    """The equipped skills as a tool. Satisfies Tool."""

    def __init__(
        self,
        runtime: SkillRuntime,
        pick: SkillPick,
        gen_length: int | None = None,
        steps: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.pick = pick
        self.gen_length = gen_length
        self.steps = steps

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=NAME,
            description=(
                f"Send text to the diffusion model with {', '.join(self.pick.equipped)} equipped "
                "and return what it generates. Give it the input the skill works on, not "
                "instructions about the skill."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The text the skill works on."}
                },
                "required": ["input"],
            },
            risk=RiskClass.READ_PUBLIC,
            source="skill",
        )

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:
        text = arguments.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("run_skill needs a non-empty input string")
        request = SkillRequest(
            prompt=text,
            skills=self.pick.skills if self.pick.schedule is None else [],
            schedule=self.pick.schedule,
            gen_length=self.gen_length,
            steps=self.steps,
        )
        try:
            result = await self.runtime.run(ctx, request)
        except SkillRuntimeError as exc:
            raise ToolError(str(exc)) from exc
        return ToolOutput(
            text=result.text,
            data={"skills": result.skills, "duration_ms": result.duration_ms},
        )
