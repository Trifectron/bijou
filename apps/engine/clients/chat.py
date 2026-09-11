"""The chat model over an OpenAI-compatible /chat/completions endpoint, llama-server by default.

The only place the provider's JSON is read or written. Transport failures, 429 and 5xx are
retryable; any other non-200 is not.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from engine.core.config import Llm
from engine.core.types.agent import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    RequestContext,
    Role,
    ToolCall,
    Usage,
)
from engine.core.types.errors import ModelError

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
}


def message_to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.role is Role.ASSISTANT and message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.role is Role.TOOL:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def to_wire(req: ModelRequest, cfg: Llm) -> dict[str, Any]:
    """The request body."""
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [message_to_wire(m) for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "chat_template_kwargs": {"enable_thinking": cfg.thinking},
    }
    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in req.tools
        ]
    if req.json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": req.json_schema},
        }
    return body


def _call(index: int, raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    name = str(function.get("name", ""))
    arguments = function.get("arguments") or "{}"
    call_id = str(raw.get("id") or f"call-{index}")
    if isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments=arguments)
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, error=exc.msg)
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, error="arguments are not an object")
    return ToolCall(id=call_id, name=name, arguments=parsed)


def from_wire(payload: dict[str, Any]) -> ModelResponse:
    """The reply. Reasoning between think tags is dropped."""
    choice = payload["choices"][0]
    message = choice["message"]
    usage = payload.get("usage") or {}
    return ModelResponse(
        content=_THINK.sub("", message.get("content") or "").strip(),
        tool_calls=[_call(i, c) for i, c in enumerate(message.get("tool_calls") or [])],
        finish_reason=_FINISH.get(str(choice.get("finish_reason")), FinishReason.OTHER),
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        ),
        model=str(payload.get("model", "")),
    )


class OpenAIChat:
    """Satisfies ChatModel."""

    def __init__(self, cfg: Llm, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        headers = {}
        if cfg.api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {cfg.api_key.get_secret_value()}"
        self.client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout_secs
        )

    async def generate(self, ctx: RequestContext, req: ModelRequest) -> ModelResponse:
        timeout = max(min(self.cfg.timeout_secs, ctx.remaining()), 0.001)
        try:
            response = await self.client.post(
                "/chat/completions", json=to_wire(req, self.cfg), timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ModelError(f"chat model timed out after {timeout:.0f}s", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ModelError(
                f"chat model at {self.cfg.base_url} unreachable: {exc}", retryable=True
            ) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise ModelError(
                f"chat model returned {response.status_code}: {response.text[:300]}",
                retryable=True,
            )
        if response.status_code != 200:
            raise ModelError(f"chat model returned {response.status_code}: {response.text[:300]}")
        try:
            return from_wire(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"chat model reply is unreadable: {exc}") from exc

    async def aclose(self) -> None:
        await self.client.aclose()
