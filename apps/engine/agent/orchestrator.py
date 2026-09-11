"""The orchestrator: one request from plan to answer, as one session.

  catalog   the skill server's trained skills; unreachable means none, with a notice
  plan      steps with dependencies
  waves     every step whose dependencies are done, planning.concurrency at a time
  per step  the selector equips skills and a subagent runs the step's tool loop
  pending   a step waiting on the user stops scheduling; confirm resumes it and the rest
  answer    one step's answer, or a synthesis of every step's result

The session is saved after the plan and after every wave. A confirmed run carries on from the
stored record.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from engine.agent.call import Stopped, complete
from engine.agent.planner import Planner
from engine.agent.policy import payload_hash
from engine.agent.selector import SkillSelector
from engine.agent.subagent import StepOutcome, Subagent
from engine.agent.toolset import ToolSet
from engine.agent.trace import Collector
from engine.core.config import AgentConfig
from engine.core.protocols import ChatModel, SessionStore, SkillRuntime, TraceSink
from engine.core.types.agent import (
    Message,
    ModelRequest,
    PlanStep,
    RequestContext,
    RunResult,
    RunStatus,
    SessionRecord,
    SkillInfo,
    StepReport,
    TraceEvent,
    TraceKind,
    now,
)
from engine.core.types.errors import ConfirmationError, ModelError, SkillRuntimeError, StoreError


class Orchestrator:
    """Plans, equips and runs steps, and answers. Owns no state beyond its collaborators."""

    def __init__(
        self,
        *,
        model: ChatModel,
        planner: Planner,
        selector: SkillSelector,
        subagent: Subagent,
        runtime: SkillRuntime,
        sessions: SessionStore,
        tools: ToolSet,
        trace: TraceSink,
        collector: Collector,
        cfg: AgentConfig,
    ) -> None:
        self.model = model
        self.planner = planner
        self.selector = selector
        self.subagent = subagent
        self.runtime = runtime
        self.sessions = sessions
        self.tools = tools
        self.trace = trace
        self.collector = collector
        self.cfg = cfg

    async def run(
        self, request: str, resume_from: str | None = None, user_id: str = "local"
    ) -> RunResult:
        """Run one request as a new session, optionally continuing an earlier one."""
        started = time.monotonic()
        context = self._prior(resume_from)
        ctx = RequestContext.start(uuid4().hex[:12], self.cfg.loop.run_timeout_secs, user_id)
        self._emit(ctx, TraceKind.RUN_STARTED, request=request, resumed_from=resume_from)
        record = SessionRecord(
            id=ctx.session_id,
            created_at=now(),
            updated_at=now(),
            request=request,
            status=RunStatus.RUNNING,
            resumed_from=resume_from,
        )
        catalog = await self._catalog(ctx)
        plan, usage = await self.planner.plan(
            ctx, request, context, self.tools.definitions(), catalog
        )
        record.plan, record.usage = plan, usage
        self._save(record)
        return await self._drive(ctx, record, catalog, context, started)

    async def confirm(self, session_id: str, token: str, approve: bool) -> RunResult:
        """Approve or decline the action a session is waiting on, then carry the run on."""
        started = time.monotonic()
        record = self.sessions.get(session_id)
        if record is None:
            raise ConfirmationError(f"no session {session_id}")
        waiting = next((p for p in record.pending if p.confirmation.token == token), None)
        if waiting is None:
            raise ConfirmationError(f"session {session_id} has no action waiting on that token")
        if waiting.confirmation.expires_at < now():
            record.pending.remove(waiting)
            self._save(record)
            raise ConfirmationError("that confirmation expired; run the request again")
        if payload_hash(waiting.call.arguments) != waiting.confirmation.payload_hash:
            raise ConfirmationError("the waiting action no longer matches what was confirmed")

        ctx = RequestContext.start(session_id, self.cfg.loop.run_timeout_secs)
        self._emit(ctx, TraceKind.CONFIRMED, tool=waiting.call.name, approved=approve)
        record.pending.remove(waiting)
        record.status = RunStatus.RUNNING
        outcome = await self.subagent.resume(ctx, waiting, approve)
        self._absorb(record, outcome)
        self._save(record)
        catalog = await self._catalog(ctx)
        return await self._drive(ctx, record, catalog, self._prior(record.resumed_from), started)

    # ---------- driving the plan ----------

    async def _drive(
        self,
        ctx: RequestContext,
        record: SessionRecord,
        catalog: list[SkillInfo],
        context: str,
        started: float,
    ) -> RunResult:
        plan = record.plan
        if plan is None:
            raise StoreError(f"session {record.id} has no plan")
        limit = asyncio.Semaphore(self.cfg.planning.concurrency)
        descriptions = {s.name: s.description for s in catalog}

        async def one(step: PlanStep, done: dict[str, StepReport]) -> StepOutcome:
            async with limit:
                return await self._step(ctx, record, step, catalog, context, done, descriptions)

        while not record.pending:
            done = {r.id: r for r in record.steps}
            ready = [
                s for s in plan.steps if s.id not in done and all(d in done for d in s.depends_on)
            ]
            if not ready:
                break
            outcomes = await asyncio.gather(*(one(s, done) for s in ready))
            for outcome in outcomes:
                self._absorb(record, outcome)
            self._save(record)

        if record.pending:
            record.status = RunStatus.AWAITING_CONFIRMATION
            record.answer = (
                f"Waiting for your confirmation: {record.pending[0].confirmation.summary}"
            )
        else:
            record.answer, record.status = await self._answer(ctx, record)
            self._emit(ctx, TraceKind.RUN_DONE, status=record.status.value)
        self._save(record)
        return RunResult(
            session_id=record.id,
            request=record.request,
            status=record.status,
            answer=record.answer,
            plan=plan,
            steps=self._ordered(record),
            usage=record.usage,
            duration_ms=int((time.monotonic() - started) * 1000),
            confirmation=record.pending[0].confirmation if record.pending else None,
            events=self.collector.take(record.id),
        )

    async def _step(
        self,
        ctx: RequestContext,
        record: SessionRecord,
        step: PlanStep,
        catalog: list[SkillInfo],
        context: str,
        done: dict[str, StepReport],
        descriptions: dict[str, str],
    ) -> StepOutcome:
        earlier = [
            f"[{d}] {done[d].goal}\nstatus: {done[d].status.value}\nresult: {done[d].answer}"
            for d in step.depends_on
        ]
        full = "\n\n".join([p for p in (context, *earlier) if p])
        try:
            pick, usage = await self.selector.pick(ctx, step, catalog)
        except Stopped as stopped:
            report = StepReport(id=step.id, goal=step.goal, status=stopped.status)
            return StepOutcome(report=report)
        outcome = await self.subagent.run(ctx, step, record.request, full, pick, descriptions)
        outcome.report.usage = outcome.report.usage.plus(usage)
        if outcome.pending is not None:
            outcome.pending.usage = outcome.report.usage
        return outcome

    def _absorb(self, record: SessionRecord, outcome: StepOutcome) -> None:
        """Keep a finished step's report, or hold a waiting one."""
        if outcome.pending is not None:
            record.pending.append(outcome.pending)
            return
        record.steps.append(outcome.report)
        record.usage = record.usage.plus(outcome.report.usage)

    def _ordered(self, record: SessionRecord) -> list[StepReport]:
        order = {s.id: i for i, s in enumerate(record.plan.steps if record.plan else [])}
        return sorted(record.steps, key=lambda r: order.get(r.id, len(order)))

    async def _answer(self, ctx: RequestContext, record: SessionRecord) -> tuple[str, RunStatus]:
        reports = self._ordered(record)
        if not reports:
            return "No step ran.", RunStatus.ERROR
        if len(reports) == 1:
            return reports[0].answer, reports[0].status
        if not any(r.status is RunStatus.ANSWERED for r in reports):
            failed = "; ".join(f"step {r.id} {r.status.value}" for r in reports)
            return f"No step finished: {failed}.", reports[0].status
        results = "\n\n".join(
            f"[{r.id}] {r.goal}\nstatus: {r.status.value}\nresult: {r.answer}" for r in reports
        )
        request = ModelRequest(
            messages=[
                Message.system(self.cfg.prompt.synthesis),
                Message.user(f"Request: {record.request}\n\nStep results:\n{results}"),
            ],
            max_tokens=self.cfg.llm.max_tokens,
            temperature=self.cfg.llm.temperature,
        )
        try:
            response = await complete(
                self.model, ctx, request, self.cfg.loop, self.trace, "", "synthesis"
            )
        except (Stopped, ModelError) as exc:
            self._emit(ctx, TraceKind.NOTICE, message=f"synthesis failed: {exc}")
            return results, RunStatus.ANSWERED
        record.usage = record.usage.plus(response.usage)
        return response.content.strip() or results, RunStatus.ANSWERED

    # ---------- helpers ----------

    async def _catalog(self, ctx: RequestContext) -> list[SkillInfo]:
        try:
            return await self.runtime.catalog()
        except SkillRuntimeError as exc:
            self._emit(ctx, TraceKind.NOTICE, message=f"running without skills: {exc}")
            return []

    def _prior(self, session_id: str | None) -> str:
        if session_id is None:
            return ""
        earlier = self.sessions.get(session_id)
        if earlier is None:
            raise StoreError(f"no session {session_id} to resume")
        text = f"Request: {earlier.request}\nAnswer: {earlier.answer}"
        return text[: self.cfg.sessions.resume_chars]

    def _save(self, record: SessionRecord) -> None:
        record.updated_at = now()
        self.sessions.save(record)

    def _emit(self, ctx: RequestContext, kind: TraceKind, **data: object) -> None:
        self.trace.emit(TraceEvent(kind=kind, session_id=ctx.session_id, data=dict(data)))
