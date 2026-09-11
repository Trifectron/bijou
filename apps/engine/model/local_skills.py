"""The skill bank in this process: the diffusion model loaded on first use, run off the loop.

The model stack is imported only when the bank is first needed, so an engine without torch still
runs the agent, with no skills and a notice saying why.
"""

from __future__ import annotations

import asyncio

from engine.core.config import Config
from engine.core.types.agent import RequestContext, SkillInfo, SkillRequest, SkillResult
from engine.core.types.errors import EngineError, SkillRuntimeError
from engine.runtime.bank import SkillBank


class LocalSkillRuntime:
    """Satisfies SkillRuntime."""

    def __init__(self, cfg: Config, bank: SkillBank | None = None) -> None:
        self.cfg = cfg
        self._bank = bank
        self._loading = asyncio.Lock()

    async def _loaded(self) -> SkillBank:
        async with self._loading:
            if self._bank is None:
                try:
                    self._bank = await asyncio.to_thread(SkillBank, self.cfg)
                except (EngineError, ImportError, OSError) as exc:
                    raise SkillRuntimeError(f"the skill bank could not load: {exc}") from exc
            return self._bank

    async def catalog(self) -> list[SkillInfo]:
        return (await self._loaded()).catalog()

    async def run(self, ctx: RequestContext, req: SkillRequest) -> SkillResult:
        bank = await self._loaded()
        timeout = max(ctx.remaining(), 0.001)
        try:
            return await asyncio.wait_for(asyncio.to_thread(bank.generate, req), timeout)
        except TimeoutError as exc:
            raise SkillRuntimeError(f"generation took longer than {timeout:.0f}s") from exc
        except SkillRuntimeError:
            raise
        except EngineError as exc:
            raise SkillRuntimeError(f"the skill bank refused the request: {exc}") from exc

    async def aclose(self) -> None:
        return None
