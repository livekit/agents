from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

from livekit.agents import llm
from livekit.plugins import vlmrun
from livekit.plugins.vlmrun import llm as vlmrun_llm

pytestmark = pytest.mark.unit

_STREAM_RESPONSE = b"""data: {"id":"c1","choices":[{"delta":{"content":"ok","role":"assistant"},"finish_reason":null,"index":0}],"created":0,"model":"qwen/qwen3.5-0.8b","object":"chat.completion.chunk"}

data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"model":"qwen/qwen3.5-0.8b","object":"chat.completion.chunk"}

data: [DONE]

"""

_PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class _Transport(httpx.AsyncBaseTransport):
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


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    t = _Transport()

    class _Client(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("limits", None)
            super().__init__(transport=t, **kwargs)

    monkeypatch.setattr(vlmrun_llm.httpx, "AsyncClient", _Client)
    monkeypatch.setenv("VLMRUN_API_KEY", "vlm-default")
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_URL", raising=False)
    return t


async def _run(model: vlmrun.LLM, content: list[Any] | None = None) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content=content or ["hi"])
    try:
        stream = model.chat(chat_ctx=chat_ctx)
        try:
            async for _ in stream:
                pass
        finally:
            await stream.aclose()
    finally:
        await model.aclose()


async def test_default_route_and_headers(transport: _Transport) -> None:
    await _run(vlmrun.LLM())

    req = transport.requests[0]
    assert req.headers["authorization"] == "Bearer vlm-default"
    assert req.headers["x-vlmrun-client"].startswith("livekit-agents/")
    assert str(req.url) == "https://gateway.vlm.run/openai/chat/completions"
    assert json.loads(req.content)["model"] == "qwen/qwen3.8-27b", "default model"


async def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VLMRUN_API_KEY is required"):
        vlmrun.LLM()


async def test_api_key_sent_as_bearer(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b", api_key="vlm-test"))

    req = transport.requests[0]
    assert req.headers["authorization"] == "Bearer vlm-test"
    assert req.headers["x-vlmrun-client"].startswith("livekit-agents/")


async def test_api_key_from_env(transport: _Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLMRUN_API_KEY", "vlm-env")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))

    assert transport.requests[0].headers["authorization"] == "Bearer vlm-env"


async def test_base_url_override(transport: _Transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://gw.example.com/")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://gw.example.com/openai/chat/completions"

    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b", base_url="https://other.example.com"))
    assert str(transport.requests[1].url) == "https://other.example.com/openai/chat/completions"


async def test_unset_sampling_params_not_sent(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="google/diffusiongemma-26b-a4b-it"))

    body = json.loads(transport.requests[0].content)
    for field in ("temperature", "top_p", "seed", "tool_choice", "parallel_tool_calls"):
        assert field not in body
    assert body["model"] == "google/diffusiongemma-26b-a4b-it"


async def test_set_sampling_params_sent(transport: _Transport) -> None:
    await _run(vlmrun.LLM(model="qwen/qwen3.8-27b", temperature=0.2, top_p=0.9))

    body = json.loads(transport.requests[0].content)
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_unsupported_sampling_param_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        vlmrun.LLM(model="google/diffusiongemma-26b-a4b-it", temperature=0.5)


async def test_image_sent_as_image_url_part(transport: _Transport) -> None:
    await _run(
        vlmrun.LLM(model="qwen/qwen3.5-0.8b"),
        content=["What is this?", llm.ImageContent(image=_PIXEL)],
    )

    body = json.loads(transport.requests[0].content)
    parts = body["messages"][-1]["content"]
    image_parts = [p for p in parts if p["type"] == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/")


def test_provider_and_model() -> None:
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", api_key="vlm-test")
    assert model.model == "qwen/qwen3.5-0.8b"
    assert model.provider == "gateway.vlm.run"


async def _messages_sent(transport: _Transport, chat_ctx: llm.ChatContext) -> list[dict[str, Any]]:
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b")
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass
    finally:
        await model.aclose()
    return json.loads(transport.requests[-1].content)["messages"]


async def test_instructions_only_request_gets_placeholder_user_message(
    transport: _Transport,
) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Greet the user.")
    messages = await _messages_sent(transport, chat_ctx)
    assert messages[-1] == {"role": "user", "content": "(empty)"}
    assert len(chat_ctx.items) == 1, "the caller's context is not modified"


async def test_existing_user_message_is_left_alone(transport: _Transport) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Be terse.")
    chat_ctx.add_message(role="user", content="hi")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["system", "user"]


async def test_later_instructions_keep_their_place(transport: _Transport) -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="You can see the camera.")
    chat_ctx.add_message(role="user", content="hi")
    chat_ctx.add_message(role="assistant", content="hello")
    chat_ctx.add_message(role="system", content="Offer to describe the view.")
    messages = await _messages_sent(transport, chat_ctx)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "You can see the camera.", "only the preamble is system"
    assert "Offer to describe the view." in messages[-1]["content"], "comes after the history"


async def test_base_url_with_sdk_v1_suffix(transport: _Transport, monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://gw.internal/v1")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://gw.internal/openai/chat/completions"


async def test_sdk_gateway_url_env(transport: _Transport, monkeypatch) -> None:
    monkeypatch.setenv("VLMRUN_GATEWAY_URL", "https://sdk.example.com/v1")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[0].url) == "https://sdk.example.com/openai/chat/completions"

    monkeypatch.setenv("VLMRUN_GATEWAY_BASE_URL", "https://base.example.com")
    await _run(vlmrun.LLM(model="qwen/qwen3.5-0.8b"))
    assert str(transport.requests[1].url) == "https://base.example.com/openai/chat/completions", (
        "VLMRUN_GATEWAY_BASE_URL wins when both are set"
    )


async def test_default_read_timeout_outlasts_gateway_first_token_deadline() -> None:
    model = vlmrun.LLM(model="qwen/qwen3.8-27b", api_key="vlm-test")
    try:
        assert model._client.timeout.read is not None and model._client.timeout.read > 60, (
            "qwen/qwen3.8-27b may take up to 60 s to its first token"
        )
    finally:
        await model.aclose()
