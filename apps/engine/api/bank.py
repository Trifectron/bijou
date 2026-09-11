"""The skill bank over HTTP, for an agent on another machine. engine serve --bank.

GET  /health    what the bank loaded
GET  /skills    SkillInfo rows
POST /generate  takes a SkillRequest, returns a SkillResult
GET  /metrics   the bank's Prometheus metrics
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CollectorRegistry

from engine.core.config import Config
from engine.core.types.agent import SkillInfo, SkillRequest, SkillResult
from engine.core.types.errors import EngineError
from engine.runtime.bank import SkillBank
from engine.telemetry.metrics import BankMetrics, exposition, new_registry


def create_app(bank: SkillBank, registry: CollectorRegistry | None = None) -> FastAPI:
    """The routes. Generation runs in the server's thread pool, one at a time."""
    app = FastAPI(title="engine skill bank", version="0.1.0")
    served = registry or new_registry()

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

    @app.get("/metrics")
    def metrics() -> Response:
        body, kind = exposition(served)
        return Response(content=body, media_type=kind)

    return app


def serve(cfg: Config) -> None:
    """Load the bank and serve it until interrupted."""
    import uvicorn

    registry = new_registry()
    bank = SkillBank(cfg, metrics=BankMetrics(registry))
    uvicorn.run(create_app(bank, registry), host=cfg.serve.host, port=cfg.serve.port)
