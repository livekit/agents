"""AMD tags its own Inference requests with X-LiveKit-Client-Feature, and nothing else."""

from __future__ import annotations

import json
from contextlib import aclosing
from typing import Any
from unittest.mock import Mock

import httpx2 as httpx
import openai
import pytest

import livekit.agents.inference.stt as inference_stt
from livekit import rtc
from livekit.agents import inference, llm
from livekit.agents.inference._utils import HEADER_CLIENT_FEATURE, HEADER_SESSION_ID
from livekit.agents.types import NOT_GIVEN, APIConnectOptions
from livekit.agents.voice.amd import _inference
from livekit.agents.voice.amd._chat_context import AMDRequest
from livekit.agents.voice.amd._stt import AMDTranscriber
from livekit.agents.voice.amd.events import AMDCategory

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import DrainingSTT

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

# Long enough to keep PyJWT's short-key warning out of the suite output.
_SECRET = "f" * 32

_CHAT_CTX = llm.ChatContext()
_CHAT_CTX.add_message(role="user", content="hello?")
_REQUEST = AMDRequest(
    stage=AMDCategory.UNCERTAIN,
    recommended_next_categories=sorted(AMDCategory),
    previous_prediction=None,
    chat_ctx=_CHAT_CTX,
    speech_duration=0.5,
)


def _sse(choices: list[dict[str, Any]]) -> bytes:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "google/gemma-4-31b-it",
        "choices": choices,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


_RECORD_RESULT = b"".join(
    [
        _sse(
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "record_result",
                                    "arguments": json.dumps({"category": "human"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        ),
        _sse([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        b"data: [DONE]\n\n",
    ]
)


def _inference_llm() -> tuple[inference.LLM, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_RECORD_RESULT
        )

    model = inference.LLM(model="google/gemma-4-31b-it", api_key=_SECRET, api_secret=_SECRET)
    model._client = openai.AsyncClient(
        api_key=_SECRET,
        base_url="http://inference.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return model, requests


@pytest.mark.asyncio
async def test_classifier_tags_only_its_own_request() -> None:
    model, requests = _inference_llm()
    model._opts.extra_kwargs["extra_headers"] = {"X-Custom": "kept"}

    assert (await _inference.classify(model, _REQUEST)).category == AMDCategory.HUMAN
    async with model.chat(chat_ctx=_CHAT_CTX) as stream:
        async for _ in stream:
            pass

    amd_request, agent_request = requests
    assert amd_request.headers[HEADER_CLIENT_FEATURE] == "AMD"
    assert amd_request.headers["X-Custom"] == "kept"
    # the agent's next turn on the same LLM is ordinary traffic
    assert HEADER_CLIENT_FEATURE not in agent_request.headers
    assert agent_request.headers["X-Custom"] == "kept"
    # the stream stamps SDK headers on a copy, never on the model's own dict
    assert model._opts.extra_kwargs["extra_headers"] == {"X-Custom": "kept"}


@pytest.mark.asyncio
async def test_plugin_llm_gets_no_header() -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="hello?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments='{"category":"human"}', call_id="r"
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    model.chat = chat  # type: ignore[method-assign]

    await _inference.classify(model, _REQUEST)

    assert chat.call_args.kwargs["extra_kwargs"] is NOT_GIVEN


def test_fallback_adapter_is_tagged_only_when_every_llm_is_inference() -> None:
    gateway = inference.LLM(model="google/gemma-4-31b-it", api_key=_SECRET, api_secret=_SECRET)
    plugin = FakeLLM()

    assert _inference._reaches_only_inference(gateway)
    assert _inference._reaches_only_inference(llm.FallbackAdapter([gateway, gateway]))
    assert not _inference._reaches_only_inference(llm.FallbackAdapter([gateway, plugin]))
    assert not _inference._reaches_only_inference(plugin)


@pytest.mark.asyncio
async def test_amd_stt_stream_is_tagged_only_for_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draining = DrainingSTT()
    gateway = inference.STT(model="cartesia/ink-whisper", api_key=_SECRET, api_secret=_SECRET)
    calls: list[dict[str, Any]] = []

    def stream(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return draining.stream(conn_options=kwargs["conn_options"])

    monkeypatch.setattr(gateway, "stream", stream)
    frame = rtc.AudioFrame.create(16000, 1, 320)

    async with aclosing(AMDTranscriber(gateway, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(frame)
    assert calls[0]["extra_headers"] == {HEADER_CLIENT_FEATURE: "AMD"}

    plugin = DrainingSTT()
    plugin_stream = Mock(wraps=plugin.stream)
    monkeypatch.setattr(plugin, "stream", plugin_stream)
    async with aclosing(AMDTranscriber(plugin, APIConnectOptions(max_retry=0))) as amd:
        amd.push_audio(frame)
    assert "extra_headers" not in plugin_stream.call_args.kwargs

    await draining.aclose()
    await plugin.aclose()
    await gateway.aclose()


@pytest.mark.asyncio
async def test_inference_stt_sends_extra_headers_below_sdk_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWebSocket:
        async def send_str(self, data: str) -> None:
            pass

    class _FakeHTTPSession:
        headers: dict[str, str] = {}

        async def ws_connect(
            self, url: str, *, headers: dict[str, str], **kwargs: object
        ) -> _FakeWebSocket:
            self.headers = headers
            return _FakeWebSocket()

    monkeypatch.setattr(
        inference_stt, "get_inference_headers", lambda: {HEADER_SESSION_ID: "inference_sdk"}
    )
    monkeypatch.setattr(inference_stt, "create_access_token", lambda *_args: "token")

    model = inference.STT(model="cartesia/ink-whisper", api_key=_SECRET, api_secret=_SECRET)
    stream = model.stream(
        extra_headers={HEADER_CLIENT_FEATURE: "AMD", HEADER_SESSION_ID: "caller_value"}
    )
    try:
        http_session = _FakeHTTPSession()
        await stream._connect_ws(http_session)  # type: ignore[arg-type]
    finally:
        await stream.aclose()
        await model.aclose()

    assert http_session.headers[HEADER_CLIENT_FEATURE] == "AMD"
    assert http_session.headers[HEADER_SESSION_ID] == "inference_sdk"
    assert http_session.headers["Authorization"] == "Bearer token"
