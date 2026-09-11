"""A subagent: one step's tool loop, with the skills its pick equipped.

Each turn is one model call. A reply with no tool calls answers the step. A reply repeating only
calls already made is told so and the next turn offers no tools; tool calls after that stall the
step. Otherwise the policy decides each call: a denied call's result says so, a call needing
confirmation stops the step in a resumable state, and allowed calls run, in parallel unless a tool
is sequential. Results are fed back as messages. The turn limit, the deadline and cancellation
stop the step.

A tool's failure is a result the model reads, never a crash of the step.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from engine.agent.llm import Stopped, complete
from engine.agent.toolset import SkillTool, ToolSet
from engine.core.config import AgentConfig
from engine.core.protocols import ChatModel, Policy, SkillRuntime, Tool, TraceSink
from engine.core.types.agent import (
    ConfirmationRequest,
    Message,
    ModelRequest,
    PendingStep,
    PlanStep,
    ProposedAction,
    RequestContext,
    RunStatus,
    SkillPick,
    StepReport,
    ToolCall,
    TraceEvent,
    TraceKind,
    Usage,
)
from engine.core.types.errors import ModelError, ToolError

REPEATED = "You already made this exact call and its result is above. Answer with what you have."
NO_TOOLS = "Tools are not available now. Answer with what you have."
STOP_TEXT = {
    RunStatus.STEP_LIMIT: "This step used every turn it had before finishing.",
    RunStatus.DEADLINE: "The run reached its deadline before this step finished.",
    RunStatus.CANCELLED: "Cancelled.",
}


def call_key(call: ToolCall) -> str:
    """Identifies a call by tool and canonical arguments, so a repeat is recognised."""
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated, {len(text) - limit} more characters]"


@dataclass
class StepOutcome:
    """How a step ended. pending is set when it waits on the user."""

    report: StepReport
    pending: PendingStep | None = None


@dataclass
class _State:
    messages: list[Message]
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    tools: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    force_answer: bool = False
    warned: bool = False


class Subagent:
    """Runs one step to an answer, a stop, or a pending confirmation."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolSet,
        runtime: SkillRuntime,
        policy: Policy,
        trace: TraceSink,
        cfg: AgentConfig,
    ) -> None:
        self.model = model
        self.tools = tools
        self.runtime = runtime
        self.policy = policy
        self.trace = trace
        self.cfg = cfg

    def toolset(self, pick: SkillPick) -> ToolSet:
        """The base tools, plus run_skill when the pick equipped anything."""
        if not pick.equipped:
            return self.tools
        skills = self.cfg.skills
        return self.tools.plus(SkillTool(self.runtime, pick, skills.gen_length, skills.steps))

    def opening(
        self,
        step: PlanStep,
        request: str,
        context: str,
        pick: SkillPick,
        descriptions: dict[str, str],
    ) -> list[Message]:
        """The system and user messages a step starts from."""
        system = self.cfg.prompt.subagent
        if pick.equipped:
            listed = "\n".join(
                f"- {n}: {descriptions.get(n, '')}".rstrip(": ") for n in pick.equipped
            )
            system += "\n\n" + self.cfg.prompt.equipped.format(skills=listed)
        user = f"Your step: {step.goal}\n\nThe whole request, for context: {request}"
        if context:
            user += f"\n\nResults of earlier steps:\n{context}"
        return [Message.system(system), Message.user(user)]

    async def run(
        self,
        ctx: RequestContext,
        step: PlanStep,
        request: str,
        context: str,
        pick: SkillPick,
        descriptions: dict[str, str] | None = None,
    ) -> StepOutcome:
        """Run a step from its opening messages."""
        state = _State(messages=self.opening(step, request, context, pick, descriptions or {}))
        return await self._loop(ctx, step, pick, state)

    async def resume(
        self, ctx: RequestContext, pending: PendingStep, approved: bool
    ) -> StepOutcome:
        """Run or decline the held call, then carry the step on from where it stopped."""
        step = PlanStep(id=pending.step_id, goal=pending.goal)
        state = _State(
            messages=list(pending.messages),
            turns=pending.turns,
            usage=pending.usage,
            tools=list(pending.tools),
            seen={call_key(pending.call)},
        )
        call = pending.call
        if approved:
            tool = self.toolset(pending.pick).get(call.name)
            if tool is None:
                content = f"error: {call.name} is no longer available"
            else:
                content = await self._execute(ctx, step.id, tool, call)
                state.tools.append(call.name)
        else:
            content = (
                "The user declined this action. Do not try it again; finish with what you have."
            )
        state.messages.append(Message.tool_result(call, content))
        return await self._loop(ctx, step, pending.pick, state)

    # ---------- the loop ----------

    async def _loop(
        self, ctx: RequestContext, step: PlanStep, pick: SkillPick, state: _State
    ) -> StepOutcome:
        tools = self.toolset(pick)
        while True:
            stop = self._limit(ctx, state)
            if stop is not None:
                return self._finish(ctx, step, pick, state, stop, STOP_TEXT[stop])
            state.turns += 1
            request = ModelRequest(
                messages=state.messages,
                tools=[] if state.force_answer else tools.definitions(),
                max_tokens=self.cfg.llm.max_tokens,
                temperature=self.cfg.llm.temperature,
            )
            try:
                response = await complete(
                    self.model, ctx, request, self.cfg.loop, self.trace, step.id, "subagent"
                )
            except Stopped as stopped:
                return self._finish(
                    ctx, step, pick, state, stopped.status, STOP_TEXT[stopped.status]
                )
            except ModelError as exc:
                return self._finish(
                    ctx, step, pick, state, RunStatus.ERROR, f"The model failed: {exc}"
                )
            state.usage = state.usage.plus(response.usage)
            state.messages.append(response.as_message())

            if not response.tool_calls:
                text = response.content.strip()
                if text:
                    return self._finish(ctx, step, pick, state, RunStatus.ANSWERED, text)
                return self._finish(
                    ctx, step, pick, state, RunStatus.STALLED, "The model returned nothing."
                )

            if state.force_answer:
                for call in response.tool_calls:
                    state.messages.append(Message.tool_result(call, NO_TOOLS))
                if state.warned:
                    return self._finish(
                        ctx, step, pick, state, RunStatus.STALLED, "The model kept calling tools."
                    )
                state.warned = True
                continue

            if all(call_key(c) in state.seen for c in response.tool_calls):
                for call in response.tool_calls:
                    state.messages.append(Message.tool_result(call, REPEATED))
                state.force_answer = True
                continue

            pending = await self._act(ctx, step, pick, tools, response.tool_calls, state)
            if pending is not None:
                report = self._report(
                    step, pick, state, RunStatus.AWAITING_CONFIRMATION, pending.confirmation.summary
                )
                return StepOutcome(report=report, pending=pending)

    def _limit(self, ctx: RequestContext, state: _State) -> RunStatus | None:
        if ctx.cancelled:
            return RunStatus.CANCELLED
        if ctx.remaining() <= 0:
            return RunStatus.DEADLINE
        if state.turns >= self.cfg.loop.max_turns:
            return RunStatus.STEP_LIMIT
        return None

    async def _act(
        self,
        ctx: RequestContext,
        step: PlanStep,
        pick: SkillPick,
        tools: ToolSet,
        calls: list[ToolCall],
        state: _State,
    ) -> PendingStep | None:
        """Authorize and run one turn's calls. Returns the held call, if one waits on the user."""
        results: dict[str, str] = {}
        runnable: list[tuple[ToolCall, Tool]] = []
        held: tuple[ToolCall, ConfirmationRequest] | None = None
        for call in calls:
            key = call_key(call)
            if key in state.seen:
                results[call.id] = REPEATED
                continue
            state.seen.add(key)
            if call.error:
                results[call.id] = f"error: the arguments are not valid JSON ({call.error})"
                continue
            tool = tools.get(call.name)
            if tool is None:
                results[call.id] = (
                    f"error: no tool named {call.name}; have {', '.join(tools.names())}"
                )
                continue
            action = ProposedAction(
                tool=call.name, arguments=call.arguments, risk=tool.definition.risk
            )
            decision = self.policy.authorize(action)
            self.trace.emit(
                TraceEvent(
                    kind=TraceKind.POLICY,
                    session_id=ctx.session_id,
                    step_id=step.id,
                    data={
                        "tool": call.name,
                        "risk": action.risk.value,
                        "decision": decision.kind,
                        "reason": decision.reason,
                    },
                )
            )
            if decision.kind == "deny":
                results[call.id] = f"denied: {decision.reason}"
            elif decision.kind == "confirm" and decision.confirmation is not None:
                if held is None:
                    held = (call, decision.confirmation)
                else:
                    results[call.id] = f"not run: {held[0].name} is waiting for the user first"
            else:
                runnable.append((call, tool))

        if any(tool.definition.sequential for _, tool in runnable):
            outputs = [await self._execute(ctx, step.id, tool, call) for call, tool in runnable]
        else:
            outputs = list(
                await asyncio.gather(
                    *(self._execute(ctx, step.id, tool, call) for call, tool in runnable)
                )
            )
        for (call, _), output in zip(runnable, outputs, strict=True):
            results[call.id] = output
            state.tools.append(call.name)

        for call in calls:
            if held is None or call.id != held[0].id:
                state.messages.append(Message.tool_result(call, results[call.id]))
        if held is None:
            return None
        return PendingStep(
            step_id=step.id,
            goal=step.goal,
            messages=list(state.messages),
            pick=pick,
            call=held[0],
            confirmation=held[1],
            turns=state.turns,
            tools=list(state.tools),
            usage=state.usage,
        )

    async def _execute(self, ctx: RequestContext, step_id: str, tool: Tool, call: ToolCall) -> str:
        """Run one call under the tool timeout. Failures come back as text."""
        shown = json.dumps(call.arguments, default=str)
        self.trace.emit(
            TraceEvent(
                kind=TraceKind.TOOL_CALL,
                session_id=ctx.session_id,
                step_id=step_id,
                data={"tool": call.name, "arguments": shown[:1000], "risk": tool.definition.risk},
            )
        )
        timeout = max(min(self.cfg.loop.tool_timeout_secs, ctx.remaining()), 0.001)
        started = time.monotonic()
        ok = False
        try:
            output = await asyncio.wait_for(tool.call(ctx, call.arguments), timeout=timeout)
            text, ok = output.text, True
        except TimeoutError:
            text = f"error: {call.name} took longer than {timeout:.0f}s"
        except ToolError as exc:
            text = f"error: {exc}"
        text = truncate(text, self.cfg.loop.max_tool_output_chars)
        self.trace.emit(
            TraceEvent(
                kind=TraceKind.TOOL_RESULT,
                session_id=ctx.session_id,
                step_id=step_id,
                data={
                    "tool": call.name,
                    "ok": ok,
                    "chars": len(text),
                    "preview": text[:300],
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        )
        return text

    def _report(
        self, step: PlanStep, pick: SkillPick, state: _State, status: RunStatus, answer: str
    ) -> StepReport:
        return StepReport(
            id=step.id,
            goal=step.goal,
            status=status,
            answer=answer,
            skills=pick.equipped,
            schedule=pick.schedule,
            pick_reason=pick.reason,
            tools=list(state.tools),
            turns=state.turns,
            usage=state.usage,
        )

    def _finish(
        self,
        ctx: RequestContext,
        step: PlanStep,
        pick: SkillPick,
        state: _State,
        status: RunStatus,
        answer: str,
    ) -> StepOutcome:
        report = self._report(step, pick, state, status, answer)
        self.trace.emit(
            TraceEvent(
                kind=TraceKind.STEP_DONE,
                session_id=ctx.session_id,
                step_id=step.id,
                data={
                    "status": status.value,
                    "turns": state.turns,
                    "skills": report.skills,
                    "tools": report.tools,
                },
            )
        )
        return StepOutcome(report=report)
