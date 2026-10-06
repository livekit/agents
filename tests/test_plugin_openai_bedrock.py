from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai
import pytest

from livekit.agents import llm
from livekit.plugins.openai import LLM

pytestmark = pytest.mark.unit


class _OneTokenStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


@llm.function_tool
async def say_hello() -> str:
    """Return a greeting."""
    return "hello"


async def _sent_tool_schema(base_url: str, tool: llm.Tool = say_hello) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_OneTokenStream(),
        )

    client = openai.AsyncClient(
        api_key="test",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    model = LLM(model="test", client=client)
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Say hello")

    try:
        async with model.chat(chat_ctx=chat_ctx, tools=[tool]) as stream:
            async for _ in stream:
                pass
    finally:
        await model.aclose()

    assert len(requests) == 1
    return requests[0]["tools"][0]["function"]["parameters"]


async def test_bedrock_mantle_empty_tool_has_required_array() -> None:
    parameters = await _sent_tool_schema("https://bedrock-mantle.us-east-1.api.aws/openai/v1")

    assert parameters["properties"] == {}
    assert parameters["required"] == []


async def test_other_openai_compatible_endpoint_preserves_empty_tool_schema() -> None:
    parameters = await _sent_tool_schema("https://compatible.example.com/v1")

    assert parameters["properties"] == {}
    assert "required" not in parameters


async def test_bedrock_mantle_does_not_mutate_a_reused_raw_tool_schema() -> None:
    @llm.function_tool(
        raw_schema={
            "name": "raw_say_hello",
            "description": "Return a greeting.",
            "parameters": {"type": "object", "properties": {}},
        }
    )
    async def raw_say_hello() -> str:
        return "hello"

    mantle_parameters = await _sent_tool_schema(
        "https://bedrock-mantle.us-east-1.api.aws/openai/v1", raw_say_hello
    )
    groq_parameters = await _sent_tool_schema("https://api.groq.com/openai/v1", raw_say_hello)

    assert mantle_parameters["required"] == []
    assert "required" not in groq_parameters
    assert "required" not in raw_say_hello.info.raw_schema["parameters"]
