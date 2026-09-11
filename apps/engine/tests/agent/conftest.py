"""Fixtures. Everything here runs in memory, with no model, no network and no GPU."""

from __future__ import annotations

import pytest

from engine.core.config import AgentConfig
from engine.core.types.agent import RequestContext


@pytest.fixture
def cfg() -> AgentConfig:
    """Defaults, with traces kept in memory and retries fast."""
    return AgentConfig(
        trace={"enabled": False},
        loop={"retry_base_ms": 1, "retry_cap_ms": 2},
    )


def tuned(cfg: AgentConfig, table: str, **values: object) -> AgentConfig:
    """cfg with some keys of one table replaced."""
    section = getattr(cfg, table).model_copy(update=values)
    return cfg.model_copy(update={table: section})


@pytest.fixture
def ctx() -> RequestContext:
    return RequestContext.start("session-1", 30)
