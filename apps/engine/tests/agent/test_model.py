"""The chat model client, OpenAI-compatible, over fake HTTP."""

import json

import httpx
import pytest

from engine.clients.chat import OpenAIChat, from_wire, to_wire
from engine.core.config import Llm
from engine.core.types.agent import (
    FinishReason,
    Message,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)
from engine.core.types.errors import ModelError


def reply(content="", tool_calls=None, finish="stop"):
    return {
        "model": "qwen",
        "choices": [
            {
                "message": {"content": content, "tool_calls": tool_calls},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def test_requests_carry_tools_schemas_and_tool_turns():
    call = ToolCall(id="c1", name="lookup", arguments={"q": "x"})
    req = ModelRequest(
        messages=[
            Message.user("hi"),
            Message.assistant("", [call]),
            Message.tool_result(call, "r"),
        ],
        tools=[ToolDefinition(name="lookup", description="d")],
        json_schema={"type": "object"},
    )
    body = to_wire(req, Llm())
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    assert body["messages"][2] == {"role": "tool", "content": "r", "tool_call_id": "c1"}
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_replies_parse_tool_calls_and_drop_reasoning():
    calls = [
        {"id": "a", "function": {"name": "t", "arguments": '{"x": 1}'}},
        {"function": {"name": "u", "arguments": "{broken"}},
        {"id": "c", "function": {"name": "v", "arguments": {"y": 2}}},
    ]
    parsed = from_wire(reply("<think>hmm</think>hello", calls, "tool_calls"))
    assert parsed.content == "hello"
    assert parsed.finish_reason is FinishReason.TOOL_CALLS
    assert parsed.tool_calls[0].arguments == {"x": 1}
    assert parsed.tool_calls[1].id == "call-1" and parsed.tool_calls[1].error
    assert parsed.tool_calls[2].arguments == {"y": 2}
    assert parsed.usage.prompt_tokens == 7


def client(handler, base="http://model/v1"):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base)


async def test_server_errors_are_retryable_and_client_errors_are_not(ctx):
    statuses = iter([503, 400])
    chat = OpenAIChat(Llm(), client(lambda r: httpx.Response(next(statuses), text="nope")))
    req = ModelRequest(messages=[Message.user("hi")])
    with pytest.raises(ModelError) as first:
        await chat.generate(ctx, req)
    assert first.value.retryable
    with pytest.raises(ModelError) as second:
        await chat.generate(ctx, req)
    assert not second.value.retryable


async def test_a_good_reply_comes_back_as_a_model_response(ctx):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, json=reply("hi there"))

    chat = OpenAIChat(Llm(model="m"), client(handler))
    response = await chat.generate(ctx, ModelRequest(messages=[Message.user("hi")]))
    assert response.content == "hi there"
    assert seen["path"] == "/v1/chat/completions" and seen["body"]["model"] == "m"
