"""The skill bank over HTTP, for an agent on another machine. engine serve-skills.

GET  /health    what the bank loaded
GET  /skills    SkillInfo rows
POST /generate  takes a SkillRequest, returns a SkillResult
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from engine.core.config import Config
from engine.core.types.agent import SkillInfo, SkillRequest, SkillResult
from engine.core.types.errors import EngineError
from engine.runtime.bank import SkillBank


def create_app(bank: SkillBank) -> FastAPI:
    """The routes. Generation runs in the server's thread pool, one at a time."""
    app = FastAPI(title="engine skill bank", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "checkpoint": bank.cfg.backend.checkpoint, "trained": len(bank.trained)}

    @app.get("/skills")
    def skills() -> list[SkillInfo]:
        return bank.catalog()

    @app.post("/generate")
    def generate(req: SkillRequest) -> SkillResult:
        try:
            return bank.generate(req)
        except EngineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def serve(cfg: Config) -> None:
    """Load the bank and serve it until interrupted."""
    import uvicorn

    uvicorn.run(create_app(SkillBank(cfg)), host=cfg.serve.host, port=cfg.serve.port)
