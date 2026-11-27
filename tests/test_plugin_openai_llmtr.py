from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

from livekit.agents.llm import ChatContext
from livekit.plugins.openai import LLM

pytestmark = pytest.mark.unit

_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_with_llmtr_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMTR_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLMTR API key is required"):
        LLM.with_llmtr()


@pytest.mark.asyncio
async def test_with_llmtr_uses_gateway_defaults() -> None:
    llm = LLM.with_llmtr(api_key="k")
    try:
        assert llm.model == "minimax/minimax-m3.1-flash"
        assert llm.provider == "llmtr.com"
        assert str(llm._client.base_url).rstrip("/") == "https://llmtr.com/v1"
    finally:
        await llm.aclose()


@pytest.mark.asyncio
async def test_with_llmtr_chat_posts_to_default_base_url() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    client = openai.AsyncClient(
        api_key="sk-llmtr-test",
        base_url="https://llmtr.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = LLM.with_llmtr(
        api_key="sk-llmtr-test",
        client=client,
        model="minimax/minimax-m3.1-flash",
    )
    ctx = ChatContext()
    ctx.add_message(role="user", content="Say ok in one word.")
    stream = llm.chat(chat_ctx=ctx)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()

    assert captured["url"].startswith("https://llmtr.com/v1/chat/completions")
    assert captured["authorization"] == "Bearer sk-llmtr-test"
    assert captured["body"]["model"] == "minimax/minimax-m3.1-flash"
