"""Cerebras plugin: LLM request encoding (gzip and msgpack) and tool history."""

from __future__ import annotations

import json
from typing import Any

import msgpack  # type: ignore[import-untyped]  # msgpack ships no type information
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import Agent, RunContext, llm
from livekit.plugins.cerebras import LLM

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]

_STREAM_RESPONSE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: [DONE]

"""


async def test_max_completion_tokens_reaches_compressed_msgpack_request() -> None:
    """https://github.com/livekit/agents/pull/6685"""
    # Catches: the Cerebras defaults (gzip and msgpack) not reaching the native request, or an
    # option missing from the encoded body.
    received: list[tuple[dict[str, str], bytes]] = []

    async def complete(request: web.Request) -> web.Response:
        headers = {name.lower(): value for name, value in request.headers.items()}
        # aiohttp undoes a gzip Content-Encoding
        received.append((headers, await request.read()))
        return web.Response(body=_STREAM_RESPONSE, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        model = LLM(
            model="gpt-oss-120b",
            api_key="test-key",
            base_url=str(server.make_url("/v1")),
            max_completion_tokens=321,
        )
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="user", content="hi")
        try:
            async with model.chat(chat_ctx=chat_ctx) as stream:
                async for _ in stream:
                    pass
        finally:
            await model.aclose()

    assert len(received) == 1
    headers, body = received[0]
    assert headers["content-type"] == "application/vnd.msgpack"
    assert headers["content-encoding"] == "gzip"
    assert headers["authorization"] == "Bearer test-key"
    payload: Wire = msgpack.unpackb(body, raw=False)
    assert payload["max_completion_tokens"] == 321


def _sse(delta: Wire, finish_reason: str) -> bytes:
    events = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-oss-120b",
            "choices": [{"index": 0, "delta": d, "finish_reason": reason}],
        }
        for d, reason in ((delta, None), ({}, finish_reason))
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (body + "data: [DONE]\n\n").encode()


async def test_tool_exchange_is_replayed_with_its_call_id_and_a_non_strict_schema() -> None:
    """Cerebras gets non-strict tools and paired call ids; https://github.com/livekit/agents/issues/1154 https://github.com/livekit/agents/issues/2403 https://github.com/livekit/agents/pull/3134."""
    received: list[Wire] = []
    call = {
        "index": 0,
        "id": "call_7",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"room": "203"}'},
    }

    async def complete(request: web.Request) -> web.Response:
        received.append(msgpack.unpackb(await request.read(), raw=False))
        if len(received) == 1:
            body = _sse({"role": "assistant", "tool_calls": [call]}, "tool_calls")
        else:
            body = _sse({"role": "assistant", "content": "Room 203 is free."}, "stop")
        return web.Response(body=body, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        model = LLM(api_key="test-key", base_url=str(server.make_url("/v1")))
        agent = Agent(instructions="Check rooms.", llm=model)

        @agent.tool
        def lookup(ctx: RunContext, room: str) -> str:
            """Check whether a room is free."""
            return f"{room} is free"

        try:
            result = await agent.run("Is room 203 free?")
        finally:
            await model.aclose()

    assert result.status == "completed"

    assert result.output == "Room 203 is free."
    assert len(received) == 2
    function = received[0]["tools"][0]["function"]
    assert function["name"] == "lookup"
    assert function.get("strict") is not True
    assert "additionalProperties" not in function["parameters"]
    history = received[1]["messages"]
    calls = [m for m in history if m["role"] == "assistant" and m.get("tool_calls")]
    results = [m for m in history if m["role"] == "tool"]
    assert [c["id"] for m in calls for c in m["tool_calls"]] == ["call_7"]
    assert [(m["tool_call_id"], m["content"]) for m in results] == [("call_7", "203 is free")]
    assert received[1]["tools"] == received[0]["tools"]
