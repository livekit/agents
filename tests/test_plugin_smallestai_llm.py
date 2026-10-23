"""Smallest AI (Electron) LLM: a thin OpenAI-compatible preset.

Electron is a drop-in OpenAI Chat Completions endpoint, so the plugin itself only
needs to own api-key resolution, the default base URL/model, and the provider label.
These tests capture the JSON body the plugin would POST, exercising the real
serializer path without network access.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx2 as httpx
import openai
import pytest

from livekit.agents.llm import ChatContext
from livekit.plugins.smallestai import LLM
from livekit.plugins.smallestai.llm import SMALLEST_LLM_BASE_URL

pytestmark = pytest.mark.unit


class _OneTokenStream(httpx.AsyncByteStream):
    """A minimal successful chat.completions SSE stream."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _llm(**kwargs: Any) -> tuple[LLM, list[dict[str, Any]], list[httpx.Request]]:
    """Build a Smallest AI LLM whose HTTP layer records every request."""
    requests: list[dict[str, Any]] = []
    raw_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw_requests.append(request)
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_OneTokenStream()
        )

    kwargs.setdefault("api_key", "test")
    client = openai.AsyncClient(
        api_key=kwargs["api_key"],
        base_url=kwargs.get("base_url", SMALLEST_LLM_BASE_URL),
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    kwargs["client"] = client
    llm_model = LLM(**kwargs)
    return llm_model, requests, raw_requests


def _plain_ctx() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=["You are a helpful assistant."])
    chat_ctx.add_message(role="user", content=["Hi!"])
    return chat_ctx


async def _run(llm_model: LLM, chat_ctx: ChatContext) -> None:
    try:
        async with llm_model.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass
    finally:
        await llm_model.aclose()


def test_default_model_and_base_url() -> None:
    llm_model, _, _ = _llm()
    assert llm_model.model == "electron"
    assert llm_model.provider == "SmallestAI"


@pytest.mark.parametrize("model", ["electron", "electron-2.0"])
async def test_model_is_sent_as_given(model: str) -> None:
    llm_model, requests, _ = _llm(model=model)
    await _run(llm_model, _plain_ctx())

    assert len(requests) == 1
    assert requests[0]["model"] == model


async def test_requests_hit_the_smallest_ai_base_url() -> None:
    llm_model, _, raw_requests = _llm()
    await _run(llm_model, _plain_ctx())

    assert len(raw_requests) == 1
    assert str(raw_requests[0].url).startswith(SMALLEST_LLM_BASE_URL)


async def test_messages_pass_through_unchanged() -> None:
    llm_model, requests, _ = _llm()
    await _run(llm_model, _plain_ctx())

    messages = cast(list[dict[str, Any]], requests[0]["messages"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1] == {"role": "user", "content": "Hi!"}


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SMALLEST_API_KEY"):
        LLM()


def test_api_key_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALLEST_API_KEY", "from-env")
    llm_model = LLM()
    assert llm_model.model == "electron"


def test_preconfigured_client_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client carrying its own credentials shouldn't need SMALLEST_API_KEY too."""
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)

    client = openai.AsyncClient(
        api_key="configured-on-client",
        base_url=SMALLEST_LLM_BASE_URL,
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ),
    )
    llm_model = LLM(client=client)
    assert llm_model.model == "electron"
