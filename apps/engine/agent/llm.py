"""Calls to the chat model: one completion, and a prompted task with a typed result.

complete makes every model call in the agent, under the run's deadline, with retries and a trace
event. structured is a task: one instruction, one call, no tools, a reply parsed into a pydantic
model; a reply that does not parse or pass the caller's check gets one correction with the error
in it, and a second failure raises PlanError. The planner and the skill selector are tasks.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from engine.core.config import Loop
from engine.core.protocols import ChatModel, TraceSink
from engine.core.types.agent import (
    Message,
    ModelRequest,
    ModelResponse,
    RequestContext,
    RunStatus,
    TraceEvent,
    TraceKind,
    Usage,
)
from engine.core.types.errors import EngineError, ModelError, PlanError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


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
        data: dict[str, object] = {
            "purpose": purpose,
            "model": response.model,
            "attempt": attempt,
            "finish_reason": response.finish_reason.value,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "tools_offered": len(req.tools),
            "tool_calls": [c.name for c in response.tool_calls],
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        keep = loop.record_content_chars
        if keep > 0:
            messages = [m.model_dump(mode="json") for m in req.messages]
            data["input"] = json.dumps(messages)[:keep]
            data["output"] = response.as_message().model_dump_json()[:keep]
        trace.emit(
            TraceEvent(
                kind=TraceKind.MODEL_CALL, session_id=ctx.session_id, step_id=step_id, data=data
            )
        )
        return response


def extract_json(text: str) -> str:
    """The outermost JSON object in a reply, without reasoning tags or code fences."""
    cleaned = _THINK.sub("", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise PlanError("the reply holds no JSON object")
    return cleaned[start : end + 1]


async def structured[T: BaseModel](
    model: ChatModel,
    ctx: RequestContext,
    *,
    schema: type[T],
    system: str,
    user: str,
    loop: Loop,
    trace: TraceSink,
    step_id: str,
    purpose: str,
    max_tokens: int,
    temperature: float,
    check: Callable[[T], None] | None = None,
) -> tuple[T, Usage]:
    """The reply parsed into schema and checked. One correction, then PlanError."""
    messages = [Message.system(system), Message.user(user)]
    usage = Usage()
    error = ""
    for _ in range(2):
        response = await complete(
            model,
            ctx,
            ModelRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=schema.model_json_schema(),
            ),
            loop,
            trace,
            step_id,
            purpose,
        )
        usage = usage.plus(response.usage)
        try:
            value = schema.model_validate_json(extract_json(response.content))
            if check is not None:
                check(value)
            return value, usage
        except (ValidationError, PlanError) as exc:
            error = str(exc)[:600]
            messages = [
                *messages,
                Message.assistant(response.content),
                Message.user(f"That reply was not valid: {error}\nReply again with only the JSON."),
            ]
    raise PlanError(f"{purpose}: {error}")
