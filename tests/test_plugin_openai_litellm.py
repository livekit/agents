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
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)


async def test_defaults_to_local_proxy() -> None:
    llm = LLM.with_litellm(model="gpt-4o", api_key="sk-1234")
    try:
        assert str(llm._client.base_url).rstrip("/") == "http://localhost:4000"
        assert llm._client.api_key == "sk-1234"
        assert llm.model == "gpt-4o"
        assert llm.provider == "localhost:4000"
        assert llm._strict_tool_schema is False
    finally:
        await llm.aclose()


async def test_reads_key_and_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-env")
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://litellm.example.com")
    llm = LLM.with_litellm(model="anthropic/claude-sonnet-4-5")
    try:
        assert llm._client.api_key == "sk-env"
        assert llm.provider == "litellm.example.com"
    finally:
        await llm.aclose()


async def test_arguments_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-env")
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://env.example.com")
    llm = LLM.with_litellm(model="gpt-4o", api_key="sk-arg", base_url="http://proxy:4000")
    try:
        assert llm._client.api_key == "sk-arg"
        assert llm.provider == "proxy:4000"
    finally:
        await llm.aclose()


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="LITELLM_PROXY_API_KEY"):
        LLM.with_litellm(model="gpt-4o")


@pytest.mark.parametrize("url", ["ftp://proxy:4000", "localhost:4000", "http://"])
def test_rejects_invalid_url(url: str) -> None:
    with pytest.raises(ValueError):
        LLM.with_litellm(model="gpt-4o", api_key="sk-1234", base_url=url)


async def test_request_reaches_proxy() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    client = openai.AsyncClient(
        api_key="sk-1234",
        base_url="http://localhost:4000",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = LLM.with_litellm(
        model="anthropic/claude-sonnet-4-5",
        api_key="sk-1234",
        client=client,
        extra_body={"metadata": {"tags": ["voice"]}},
    )
    ctx = ChatContext()
    ctx.add_message(role="user", content="Hello")
    stream = llm.chat(chat_ctx=ctx)
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()

    assert captured["url"] == "http://localhost:4000/chat/completions"
    assert captured["auth"] == "Bearer sk-1234"
    assert captured["body"]["model"] == "anthropic/claude-sonnet-4-5"
    assert captured["body"]["metadata"] == {"tags": ["voice"]}
