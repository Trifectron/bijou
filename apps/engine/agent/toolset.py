"""The tools one subagent can see, and run_skill, the tool its equipped skills are used through."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from engine.core.protocols import SkillRuntime, Tool
from engine.core.types.agent import (
    RequestContext,
    RiskClass,
    SkillPick,
    SkillRequest,
    ToolDefinition,
    ToolOutput,
)
from engine.core.types.errors import ConfigError, SkillRuntimeError, ToolError

RUN_SKILL = "run_skill"


class ToolSet:
    """An immutable, name-keyed set of tools. Two tools cannot share a name."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in self._tools:
                raise ConfigError(f"two tools are named {name}")
            self._tools[name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def plus(self, *tools: Tool) -> ToolSet:
        """A new set with these tools added."""
        return ToolSet([*self._tools.values(), *tools])

    def without(self, names: Iterable[str]) -> ToolSet:
        """A new set with the named tools removed."""
        dropped = set(names)
        return ToolSet([t for n, t in self._tools.items() if n not in dropped])

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]


class SkillTool:
    """The equipped skills as a tool. Sends its input to the diffusion model with exactly the
    pick live: the listed skills, or the phase schedule over them. Satisfies Tool."""

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
            name=RUN_SKILL,
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
            data={"skills": result.skills, "generate_ms": result.duration_ms},
        )
