"""A prompted task: one instruction, one model call, no tools, a typed result.

The planner and the skill selector are tasks. Structured output is requested with the result's
JSON schema; a reply that does not parse or does not pass the caller's check gets one correction
with the error in it, and a second failure raises PlanError.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from engine.agent.call import complete
from engine.core.config import Loop
from engine.core.protocols import ChatModel, TraceSink
from engine.core.types.agent import Message, ModelRequest, RequestContext, Usage
from engine.core.types.errors import PlanError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


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
