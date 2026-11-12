"""Sarvam LLM against a local OpenAI-compatible chat completions peer."""

from __future__ import annotations

import json
from typing import TypedDict

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import llm
from livekit.plugins import sarvam

_REPLY = (
    "data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "sarvam-105b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Sunny."},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    + "\n\ndata: [DONE]\n\n"
)


class ChatMessage(TypedDict):
    role: str


class ChatRequest(TypedDict, total=False):
    messages: list[ChatMessage]
    tools: list[object]


async def test_tool_history_is_stripped_when_the_turn_declares_no_tools() -> None:
    """Sarvam 400s on tool messages without a `tools` array, poisoning the session.

    https://github.com/livekit/agents/pull/7382
    """
    received: list[ChatRequest] = []

    async def complete(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.Response(text=_REPLY, content_type="text/event-stream")

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="weather?")
    chat_ctx.insert(
        [
            llm.FunctionCall(call_id="call_1", name="get_weather", arguments='{"city": "Pune"}'),
            llm.FunctionCallOutput(
                call_id="call_1", name="get_weather", output="sunny", is_error=False
            ),
        ]
    )
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        model = sarvam.LLM(api_key="sk_test", base_url=str(server.make_url("/v1")))
        async with model.chat(chat_ctx=chat_ctx) as stream:
            text = "".join([chunk.delta.content or "" async for chunk in stream if chunk.delta])
        await model.aclose()

    assert text == "Sunny."
    messages = received[0]["messages"]
    assert [m["role"] for m in messages] == ["user"]
    assert "tools" not in received[0]
