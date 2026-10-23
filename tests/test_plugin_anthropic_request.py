from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ToolChoice,
    function_tool,
)
from livekit.plugins.anthropic import LLM

pytestmark = pytest.mark.unit

_STREAM_RESPONSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_test","type":"message",'
    b'"role":"assistant","model":"claude","content":[],"stop_reason":null,'
    b'"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


@function_tool
async def get_weather(city: str) -> str:
    """Look up the weather."""
    return city


_NAMED: ToolChoice = {"type": "function", "function": {"name": "get_weather"}}


def _make_llm(model: str, requests: list[dict[str, Any]], **kwargs: Any) -> LLM:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_STREAM_RESPONSE
        )

    client = anthropic.AsyncClient(
        api_key="sk-ant-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return LLM(model=model, api_key="sk-ant-test", client=client, **kwargs)


async def _tool_choice(model: str, tool_choice: ToolChoice, **kwargs: Any) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    instance = _make_llm(model, requests)
    await instance.chat(
        chat_ctx=ChatContext(), tools=[get_weather], tool_choice=tool_choice, **kwargs
    ).collect()
    return requests[0]["tool_choice"]


async def test_forced_tool_choice_sent_as_auto_only_for_rejecting_models() -> None:
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
        assert await _tool_choice(model, "required") == {"type": "auto"}
        assert await _tool_choice(model, _NAMED) == {"type": "auto"}

    assert await _tool_choice("claude-sonnet-5-5", "required", parallel_tool_calls=False) == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }

    # Opus 5, Sonnet 5, Fable 5 and Haiku 5.5 still accept a forced choice.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-5-5"):
        assert await _tool_choice(model, "required") == {"type": "any"}
        assert await _tool_choice(model, _NAMED) == {"type": "tool", "name": "get_weather"}


async def test_forced_tool_choice_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[dict[str, Any]] = []
    instance = _make_llm("claude-sonnet-5-5", requests, tool_choice="required")
    with caplog.at_level("WARNING"):
        for _ in range(3):
            await instance.chat(chat_ctx=ChatContext(), tools=[get_weather]).collect()

    warnings = [r for r in caplog.records if "forced tool_choice" in r.getMessage()]
    assert len(warnings) == 1
    assert [r["tool_choice"] for r in requests] == [{"type": "auto"}] * 3


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


_TOOL_STEP_RESPONSE = _sse(
    {
        "type": "message_start",
        "message": {
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Check "},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Paris."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "redacted_thinking", "data": "opaque"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
)

_THINKING = [
    {"type": "thinking", "thinking": "Check Paris.", "signature": "sig"},
    {"type": "redacted_thinking", "data": "opaque"},
]


async def _after_tool_step(**kwargs: Any) -> tuple[LLM, list[dict[str, Any]], ChatContext]:
    """Run a tool step whose response thinks first, and add the call and its result to the history."""
    requests: list[dict[str, Any]] = []
    responses = iter([_TOOL_STEP_RESPONSE])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = next(responses, _STREAM_RESPONSE)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = anthropic.AsyncClient(
        api_key="sk-ant-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    instance = LLM(model="claude-haiku-5-5", api_key="sk-ant-test", client=client, **kwargs)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="You check the weather.")
    chat_ctx.add_message(role="user", content="Weather in Paris?")
    response = await instance.chat(chat_ctx=chat_ctx, tools=[get_weather]).collect()
    [call] = response.tool_calls
    chat_ctx.items.append(
        FunctionCall(
            call_id=call.call_id, name=call.name, arguments=call.arguments, extra=call.extra or {}
        )
    )
    chat_ctx.items.append(
        FunctionCallOutput(call_id=call.call_id, name=call.name, output="Sunny.", is_error=False)
    )
    return instance, requests, chat_ctx


async def test_thinking_blocks_go_back_before_their_tool_use() -> None:
    # caching marks the last block of each turn; the thinking blocks must stay first and unmarked
    instance, requests, chat_ctx = await _after_tool_step(caching="ephemeral")
    for _ in range(2):  # every later request sends them once, unchanged
        await instance.chat(chat_ctx=chat_ctx, tools=[get_weather]).collect()

    for request in requests[1:]:
        assistant = [m["content"] for m in request["messages"] if m["role"] == "assistant"]
        assert [[c["type"] for c in content] for content in assistant] == [
            ["thinking", "redacted_thinking", "tool_use"]
        ]
        assert assistant[0][:2] == _THINKING


async def test_thinking_blocks_are_dropped_after_the_instructions_or_tools_change() -> None:
    # The API rejects a block whose system prompt or tools changed since it was made.
    instance, requests, chat_ctx = await _after_tool_step()
    edited = chat_ctx.copy()
    edited.items[0] = ChatMessage(role="system", content=["You check the weather. Be brief."])
    await instance.chat(chat_ctx=edited, tools=[get_weather]).collect()

    @function_tool
    async def get_time(city: str) -> str:
        """Look up the time."""
        return city

    await instance.chat(chat_ctx=chat_ctx, tools=[get_weather, get_time]).collect()

    for request in requests[1:]:
        types = [c["type"] for m in request["messages"] for c in m["content"]]
        assert "thinking" not in types and "redacted_thinking" not in types
