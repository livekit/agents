from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import openai
import pytest

from livekit.agents.llm import ChatContext
from livekit.plugins.openai import LLM

pytestmark = pytest.mark.unit

_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
_STREAM_RESPONSE = b"""data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"meta-llama/Llama-3.3-70B-Instruct","object":"chat.completion.chunk"}

data: {"id":"chatcmpl-test","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"meta-llama/Llama-3.3-70B-Instruct","object":"chat.completion.chunk"}

data: [DONE]

"""


class _RecordingTransport(httpx.AsyncBaseTransport):
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


def test_nebius_helper_configures_token_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "env-key")

    with patch("livekit.plugins.openai.llm.openai.AsyncClient") as client_cls:
        client = client_cls.return_value
        llm = LLM.with_nebius(model=_MODEL)

    assert llm.model == _MODEL
    assert client_cls.call_args.kwargs["api_key"] == "env-key"
    assert str(client_cls.call_args.kwargs["base_url"]) == _BASE_URL
    assert client is llm._client


async def test_nebius_helper_sends_request_to_configured_client() -> None:
    transport = _RecordingTransport()
    llm = LLM.with_nebius(
        model=_MODEL,
        api_key="test-key",
        base_url=_BASE_URL,
        client=None,
    )
    await llm._client.close()
    llm._client = openai.AsyncClient(
        api_key="test-key",
        base_url=_BASE_URL,
        http_client=httpx.AsyncClient(transport=transport),
    )
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    stream = llm.chat(chat_ctx=chat_ctx)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == f"{_BASE_URL}chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == _MODEL
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
