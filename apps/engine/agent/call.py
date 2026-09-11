"""One chat completion under the run's deadline, with retries and a trace event.

Every model call in the harness goes through complete, so every stopping condition a model call
can hit, the deadline, cancellation and a retryable failure, is handled in one place.
"""

from __future__ import annotations

import asyncio
import time

from engine.core.config import Loop
from engine.core.protocols import ChatModel, TraceSink
from engine.core.types.agent import (
    ModelRequest,
    ModelResponse,
    RequestContext,
    RunStatus,
    TraceEvent,
    TraceKind,
)
from engine.core.types.errors import EngineError, ModelError


class Stopped(EngineError):
    """The run hit its deadline or was cancelled while waiting on the model."""

    def __init__(self, status: RunStatus) -> None:
        super().__init__(status.value)
        self.status = status


def backoff(attempt: int, base_ms: int, cap_ms: int, remaining: float) -> float:
    """Seconds to wait before retry number attempt, never past the deadline."""
    delay = min(cap_ms, base_ms * (1 << max(attempt - 1, 0))) / 1000
    return max(min(delay, remaining), 0.0)


async def complete(
    model: ChatModel,
    ctx: RequestContext,
    req: ModelRequest,
    loop: Loop,
    trace: TraceSink,
    step_id: str,
    purpose: str,
) -> ModelResponse:
    """The model's reply. Raises Stopped, or the last ModelError once retries are spent."""
    attempt = 0
    while True:
        if ctx.cancelled:
            raise Stopped(RunStatus.CANCELLED)
        remaining = ctx.remaining()
        if remaining <= 0:
            raise Stopped(RunStatus.DEADLINE)
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(model.generate(ctx, req), timeout=remaining)
        except TimeoutError as exc:
            raise Stopped(RunStatus.DEADLINE) from exc
        except ModelError as exc:
            retry = exc.retryable and attempt < loop.max_model_retries
            trace.emit(
                TraceEvent(
                    kind=TraceKind.MODEL_ERROR,
                    session_id=ctx.session_id,
                    step_id=step_id,
                    data={
                        "purpose": purpose,
                        "attempt": attempt,
                        "error": str(exc),
                        "retried": retry,
                    },
                )
            )
            if not retry:
                raise
            attempt += 1
            await asyncio.sleep(
                backoff(attempt, loop.retry_base_ms, loop.retry_cap_ms, ctx.remaining())
            )
            continue
        trace.emit(
            TraceEvent(
                kind=TraceKind.MODEL_CALL,
                session_id=ctx.session_id,
                step_id=step_id,
                data={
                    "purpose": purpose,
                    "model": response.model,
                    "attempt": attempt,
                    "finish_reason": response.finish_reason.value,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "tools_offered": len(req.tools),
                    "tool_calls": [c.name for c in response.tool_calls],
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
        )
        return response
