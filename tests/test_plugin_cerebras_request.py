from __future__ import annotations

import gzip
import inspect
from typing import Any

import httpx
import msgpack
import pytest

from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN
from livekit.plugins.cerebras import LLM
from livekit.plugins.cerebras.llm import _CerebrasClient

pytestmark = pytest.mark.unit


_STREAM_RESPONSE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"gpt-oss-120b","object":"chat.completion.chunk"}

data: [DONE]

"""


class _MockChatCompletionsTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_STREAM_RESPONSE,
            request=request,
        )


async def _capture_chat_request(max_completion_tokens: int | None = None) -> httpx.Request:
    transport = _MockChatCompletionsTransport()
    client = _CerebrasClient(
        use_gzip=True,
        use_msgpack=True,
        api_key="test-key",
        base_url="https://api.cerebras.ai/v1",
        http_client=httpx.AsyncClient(transport=transport),
    )
    if max_completion_tokens is None:
        model = LLM(model="gpt-oss-120b", api_key="test-key", client=client)
    else:
        model = LLM(
            model="gpt-oss-120b",
            api_key="test-key",
            client=client,
            max_completion_tokens=max_completion_tokens,
        )

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    try:
        stream = model.chat(chat_ctx=chat_ctx)
        try:
            async for _ in stream:
                pass
        finally:
            await stream.aclose()
    finally:
        await model.aclose()
        await client.close()

    assert len(transport.requests) == 1
    return transport.requests[0]


def _request_payload(request: httpx.Request) -> dict[str, Any]:
    body = gzip.decompress(request.content)
    payload = msgpack.unpackb(body, raw=False)
    assert isinstance(payload, dict)
    return payload


def test_constructor_exposes_max_completion_tokens() -> None:
    parameter = inspect.signature(LLM.__init__).parameters["max_completion_tokens"]
    assert parameter.default is NOT_GIVEN


@pytest.mark.asyncio
async def test_max_completion_tokens_reaches_compressed_msgpack_request() -> None:
    request = await _capture_chat_request(max_completion_tokens=321)

    assert request.headers["content-type"] == "application/vnd.msgpack"
    assert request.headers["content-encoding"] == "gzip"
    assert _request_payload(request)["max_completion_tokens"] == 321


@pytest.mark.asyncio
async def test_omitted_max_completion_tokens_does_not_change_request() -> None:
    request_without_limit = await _capture_chat_request()
    request_with_limit = await _capture_chat_request(max_completion_tokens=321)
    payload_without_limit = _request_payload(request_without_limit)
    payload_with_limit = _request_payload(request_with_limit)

    assert "max_completion_tokens" not in payload_without_limit
    assert payload_with_limit.pop("max_completion_tokens") == 321
    assert payload_with_limit == payload_without_limit
