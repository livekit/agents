from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web

from livekit.agents import APIStatusError, function_tool
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.llm._replay import ENV_LLM_REPLAY_URL

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


@function_tool
async def lookup_caller(phone: str) -> str:
    """Look up a caller's record by phone number."""
    return "{}"


@pytest.fixture
async def replay_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    state: dict[str, Any] = {"requests": [], "status": 200, "body": {}}

    async def handle(request: web.Request) -> web.Response:
        state["requests"].append(await request.json())
        return web.json_response(state["body"], status=state["status"])

    app = web.Application()
    app.router.add_post("/llm", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = site._server.sockets[0].getsockname()[:2]  # type: ignore[union-attr]
    monkeypatch.setenv(ENV_LLM_REPLAY_URL, f"http://{host}:{port}/llm")
    try:
        yield state
    finally:
        await runner.cleanup()


def _chat_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content="You are Sam.")
    chat_ctx.add_message(role="user", content="+1 415 555 0142")
    return chat_ctx


async def test_replay_serves_the_recorded_output_instead_of_the_provider(
    replay_server: dict[str, Any],
) -> None:
    replay_server["body"] = {
        "response_id": "chatcmpl-1",
        "output_messages": [
            {
                "role": "assistant",
                "finish_reason": "tool_call",
                "parts": [
                    {"type": "text", "content": "Let me look that up."},
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "lookup_caller",
                        "arguments": {"phone": "+1 415 555 0142"},
                    },
                ],
            }
        ],
    }
    # FakeLLM has no responses, so anything streamed came from the replay server
    llm = FakeLLM()
    async with llm.chat(chat_ctx=_chat_ctx(), tools=[lookup_caller]) as stream:
        chunks = [chunk async for chunk in stream]

    assert replay_server["requests"] == [
        {
            "model": llm.model,
            "provider": "unknown",
            "system_instructions": [{"type": "text", "content": "You are Sam."}],
            "input_messages": [
                {"role": "user", "parts": [{"type": "text", "content": "+1 415 555 0142"}]}
            ],
            "tool_definitions": [
                {
                    "type": "function",
                    "name": "lookup_caller",
                    "description": "Look up a caller's record by phone number.",
                }
            ],
        }
    ]
    assert len(chunks) == 1
    assert chunks[0].id == "chatcmpl-1"
    assert chunks[0].delta is not None
    assert chunks[0].delta.content == "Let me look that up."
    assert chunks[0].delta.tool_calls == [
        FunctionToolCall(
            name="lookup_caller",
            call_id="call_1",
            arguments=json.dumps({"phone": "+1 415 555 0142"}),
        )
    ]


async def test_replay_raises_when_the_request_diverges(replay_server: dict[str, Any]) -> None:
    replay_server["status"] = 409
    replay_server["body"] = {"error": "LLM request 1 of 3 diverged from the session"}

    with pytest.raises(APIStatusError, match="diverged from the session") as exc:
        async with FakeLLM().chat(chat_ctx=_chat_ctx()) as stream:
            async for _ in stream:
                pass
    assert exc.value.status_code == 409
    assert exc.value.retryable is False
    assert len(replay_server["requests"]) == 1
