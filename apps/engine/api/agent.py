"""The agent over HTTP, engine serve. apps/evals and any other client speak to it only here.

GET  /health            tools, and whether the skill bank answers
POST /run               takes request and resume_from, returns RunResult
POST /confirm           takes session_id, token and approve, returns RunResult
GET  /sessions?q=&limit returns SessionSummary rows
GET  /sessions/{id}     returns one SessionRecord
GET  /skills            returns SkillInfo rows
GET  /patterns          returns SkillProposal rows
GET  /metrics           the agent's and the in-process bank's Prometheus metrics
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from engine.core.config import Config
from engine.core.types.agent import (
    RunResult,
    SessionRecord,
    SessionSummary,
    SkillInfo,
    SkillProposal,
)
from engine.core.types.errors import ConfirmationError, EngineError, SkillRuntimeError, StoreError
from engine.telemetry.metrics import exposition
from engine.wiring import Agent, open_agent

Factory = Callable[[Config], AbstractAsyncContextManager[Agent]]


class RunBody(BaseModel):
    request: str = Field(min_length=1)
    resume_from: str | None = None
    user_id: str = "local"


class ConfirmBody(BaseModel):
    session_id: str
    token: str
    approve: bool = True


def create_app(cfg: Config, factory: Factory = open_agent) -> FastAPI:
    """The routes over one harness, opened for the app's lifetime."""
    holder: dict[str, Agent] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with factory(cfg) as harness:
            holder["harness"] = harness
            yield
            holder.clear()

    app = FastAPI(title="engine", version="0.1.0", lifespan=lifespan)

    def current() -> Agent:
        return holder["harness"]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        harness = current()
        try:
            skills = len([s for s in await harness.runtime.catalog() if s.trained])
            skill_server = True
        except SkillRuntimeError:
            skills, skill_server = 0, False
        return {
            "ok": True,
            "tools": harness.tools.names(),
            "skill_server": skill_server,
            "trained_skills": skills,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        body, kind = exposition(current().registry)
        return Response(content=body, media_type=kind)

    @app.post("/run")
    async def run(body: RunBody) -> RunResult:
        try:
            return await current().orchestrator.run(body.request, body.resume_from, body.user_id)
        except StoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EngineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/confirm")
    async def confirm(body: ConfirmBody) -> RunResult:
        try:
            return await current().orchestrator.confirm(body.session_id, body.token, body.approve)
        except ConfirmationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/sessions")
    async def sessions(q: str = "", limit: int = 20) -> list[SessionSummary]:
        store = current().sessions
        return store.search(q, limit) if q else store.recent(limit)

    @app.get("/sessions/{session_id}")
    async def session(session_id: str) -> SessionRecord:
        record = current().sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id}")
        return record

    @app.get("/skills")
    async def skills() -> list[SkillInfo]:
        try:
            return await current().runtime.catalog()
        except SkillRuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/patterns")
    async def patterns() -> list[SkillProposal]:
        harness = current()
        try:
            existing = {s.name for s in await harness.runtime.catalog()}
        except SkillRuntimeError:
            existing = set()
        return harness.miner.propose(existing)

    return app
