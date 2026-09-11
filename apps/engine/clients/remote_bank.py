"""The diffusion model with its LoRA bank, over bijou serve's HTTP surface.

The harness keeps its own copy of the contract, apps/bijou/serve/wire.py on the other side:

  GET  /skills    returns rows of name, description, trained
  POST /generate  takes prompt, skills, schedule, instruct, gen_length, steps
                  returns text, skills, gen_length, steps, duration_ms
"""

from __future__ import annotations

from typing import Any

import httpx

from engine.core.config import SkillServer
from engine.core.types.agent import RequestContext, SkillInfo, SkillRequest, SkillResult
from engine.core.types.errors import SkillRuntimeError


class RemoteBank:
    """Satisfies SkillRuntime."""

    def __init__(self, cfg: SkillServer, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or httpx.AsyncClient(
            base_url=cfg.url.rstrip("/"), timeout=cfg.timeout_secs
        )

    def _unreachable(self, exc: Exception) -> SkillRuntimeError:
        return SkillRuntimeError(
            f"skill server at {self.cfg.url} unreachable ({exc}); start it with: just serve"
        )

    async def catalog(self) -> list[SkillInfo]:
        try:
            response = await self.client.get("/skills", timeout=10.0)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        if response.status_code != 200:
            raise SkillRuntimeError(f"skill server returned {response.status_code} for /skills")
        return [SkillInfo.model_validate(item) for item in response.json()]

    async def run(self, ctx: RequestContext, req: SkillRequest) -> SkillResult:
        body: dict[str, Any] = req.model_dump(exclude_none=True)
        timeout = max(min(self.cfg.timeout_secs, ctx.remaining()), 0.001)
        try:
            response = await self.client.post("/generate", json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise SkillRuntimeError(f"skill server took longer than {timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        if response.status_code == 422:
            detail = response.json().get("detail", response.text)
            raise SkillRuntimeError(f"skill server refused the request: {detail}")
        if response.status_code != 200:
            raise SkillRuntimeError(f"skill server returned {response.status_code}")
        return SkillResult.model_validate(response.json())

    async def aclose(self) -> None:
        await self.client.aclose()
