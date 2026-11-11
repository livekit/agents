from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import httpx2
import openai
import pytest

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.agents.llm import CacheBreakpoint, ChatContext
from livekit.agents.llm.chat_context import Instructions
from livekit.agents.voice.generation import mark_instructions_cache_boundary, update_instructions
from livekit.plugins.openai import LLM, TTS, responses

pytestmark = pytest.mark.unit

STATIC = "You are the Riverside Clinic voice agent. Follow the clinic rules."
DYNAMIC = "Current time: 09:01. Caller number: +15551234567."
DOC = "Insurance card on file: Acme Health, member 1234."
BREAKPOINT = {"mode": "explicit"}
AZURE = {"azure_endpoint": "https://res.openai.azure.com", "api_version": "2025-04-01-preview"}
PCM = b"\x00\x01" * 4800  # 200ms of 16-bit mono at 24kHz

_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _chat_ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="system", content=[STATIC, CacheBreakpoint(), DYNAMIC])
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


@pytest.mark.parametrize(
    ("make_llm", "enabled"),
    [
        (lambda: LLM(model="gpt-5.6", api_key="k"), True),
        (lambda: LLM(model="gpt-4.1", api_key="k"), False),
        (lambda: LLM(model="gpt-5.6", api_key="k", base_url="https://openrouter.ai/api/v1"), False),
        (lambda: LLM.with_azure(model="gpt-5.6", api_key="k", **AZURE), False),
        (
            lambda: LLM.with_azure(
                model="gpt-5.6", api_key="k", prompt_cache_breakpoints=True, **AZURE
            ),
            True,
        ),
        (lambda: LLM(model="gpt-5.6", api_key="k", prompt_cache_breakpoints=False), False),
        (
            lambda: LLM(
                model="gpt-4.1",
                api_key="k",
                base_url="https://gw.example/v1",
                prompt_cache_breakpoints=True,
            ),
            True,
        ),
    ],
    ids=[
        "gpt-5.6",
        "gpt-4.1",
        "compatible-host",
        "azure",
        "azure-opt-in",
        "explicit-false",
        "explicit-true",
    ],
)
async def test_chat_cache_breakpoints_resolve_per_model_and_host(make_llm, enabled: bool) -> None:
    llm = make_llm()
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        assert stream._prompt_cache_breakpoints is enabled
    finally:
        await stream.aclose()
        await llm.aclose()


async def _chat_wire_body(model: str, **kwargs: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE)

    client = openai.AsyncClient(
        api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    llm = LLM(model=model, client=client, **kwargs)
    stream = llm.chat(chat_ctx=_chat_ctx())
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()
    return captured


async def test_chat_wire_carries_breakpoint_and_cache_options_for_gpt_5_6() -> None:
    body = await _chat_wire_body("gpt-5.6", prompt_cache_options={"ttl": "30m"})

    assert body["messages"][0]["content"] == [
        {"type": "text", "text": STATIC, "prompt_cache_breakpoint": BREAKPOINT},
        {"type": "text", "text": f"\n{DYNAMIC}"},
    ]
    assert body["prompt_cache_options"] == {"ttl": "30m"}


async def test_chat_wire_omits_breakpoint_for_gpt_4_1() -> None:
    body = await _chat_wire_body("gpt-4.1")

    assert body["messages"][0]["content"] == f"{STATIC}\n{DYNAMIC}"
    assert "prompt_cache_breakpoint" not in json.dumps(body)


@pytest.mark.parametrize(
    ("kwargs", "enabled"),
    [
        ({"model": "gpt-5.6-luna"}, True),
        ({"model": "gpt-6-sol", "use_websocket": False}, True),
        ({"model": "gpt-5.6", "base_url": "wss://api.openai.com:443/v1/responses"}, True),
        (
            {
                "model": "gpt-5.6",
                "use_websocket": False,
                "base_url": "https://api.openai.com:443/v1",
            },
            True,
        ),
        ({"model": "gpt-5.6", "base_url": "wss://api.openai.com./v1/responses"}, True),
        ({"model": "gpt-4.1"}, False),
        ({"model": "gpt-5.6", "base_url": "wss://gw.example/v1/responses"}, False),
        ({"model": "gpt-5.6", "prompt_cache_breakpoints": False}, False),
        (
            {
                "model": "gpt-4.1",
                "base_url": "wss://gw.example/v1/responses",
                "prompt_cache_breakpoints": True,
            },
            True,
        ),
    ],
)
async def test_responses_cache_breakpoints_resolve_per_model_and_host(
    kwargs: dict[str, Any], enabled: bool
) -> None:
    llm = responses.LLM(api_key="k", **kwargs)
    try:
        assert llm._resolve_prompt_cache_breakpoints() is enabled
    finally:
        await llm.aclose()


def _responses_ctx() -> ChatContext:
    ctx = ChatContext()
    update_instructions(
        ctx, instructions=Instructions(STATIC, dynamic=DYNAMIC), add_if_missing=True
    )
    mark_instructions_cache_boundary(ctx)
    ctx.add_message(role="user", content="Hi, I need to reschedule.")
    return ctx


async def _responses_body(
    model: str, ctx: ChatContext | None = None, **kwargs: Any
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(500, json={"error": {"message": "stop here"}})

    client = openai.AsyncClient(
        api_key="k",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = responses.LLM(model=model, client=client, use_websocket=False, **kwargs)
    stream = llm.chat(chat_ctx=ctx or _responses_ctx(), conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIStatusError):
            async for _ in stream:
                pass
    finally:
        await stream.aclose()
        await llm.aclose()
        await client.close()
    return captured


def _tagged_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for item in body["input"]
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if "prompt_cache_breakpoint" in part
    ]


async def test_responses_wire_tags_the_instructions_for_gpt_5_6() -> None:
    body = await _responses_body("gpt-5.6-luna")

    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": STATIC, "prompt_cache_breakpoint": BREAKPOINT}
    ]
    assert body["input"][1] == {"role": "system", "content": DYNAMIC}
    assert "prompt_cache_options" not in body


async def test_responses_wire_folds_dynamic_and_omits_breakpoints_for_gpt_4_1() -> None:
    body = await _responses_body("gpt-4.1")

    assert body["input"][0] == {"role": "system", "content": f"{STATIC}\n{DYNAMIC}"}
    assert _tagged_parts(body) == []


async def test_responses_wire_tags_a_caller_placed_breakpoint_in_a_user_message() -> None:
    ctx = _responses_ctx()
    ctx.add_message(role="user", content=[DOC, CacheBreakpoint(), "Is my plan accepted?"])

    body = await _responses_body("gpt-5.6-luna", ctx=ctx)

    assert body["input"][-1]["content"] == [
        {"type": "input_text", "text": DOC, "prompt_cache_breakpoint": BREAKPOINT},
        {"type": "input_text", "text": "\nIs my plan accepted?"},
    ]


async def test_responses_wire_forwards_cache_options_even_with_breakpoints_off() -> None:
    # prompt_cache_options is the caller's own setting; it is not gated like breakpoints
    body = await _responses_body(
        "gpt-5.6-luna",
        prompt_cache_breakpoints=False,
        prompt_cache_options={"mode": "implicit", "ttl": "30m"},
    )

    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert _tagged_parts(body) == []


class _FakeWebSocket:
    async def send_json(self, _data: object) -> None:
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.url = ""

    async def ws_connect(self, url: str, **_kwargs: object) -> _FakeWebSocket:
        self.url = url
        return _FakeWebSocket()


@pytest.mark.parametrize(
    ("base_url", "model", "netloc", "query"),
    [
        (
            "wss://gateway.example.com/v1/responses",
            "gateway-responses",
            "gateway.example.com",
            {"model": ["gateway-responses"]},
        ),
        (
            "https://gateway.example.com/v1/responses",
            "gateway-responses",
            "gateway.example.com",
            {"model": ["gateway-responses"]},
        ),
        (None, "gpt-4.1", "api.openai.com", {}),
    ],
    ids=["gateway", "gateway-http-scheme", "native"],
)
async def test_responses_connect_url_names_the_model_only_for_a_gateway(
    base_url: str | None, model: str, netloc: str, query: dict[str, list[str]]
) -> None:
    kwargs = {"base_url": base_url} if base_url else {}
    instance = responses.LLM(api_key="test-key", model=model, use_websocket=True, **kwargs)
    assert instance._ws is not None
    session = _FakeSession()
    instance._ws._session = session

    await instance._ws._create_ws(5)

    url = urlparse(session.url)
    assert (url.scheme, url.netloc, url.path) == ("wss", netloc, "/v1/responses")
    assert parse_qs(url.query) == query


def _tts(handler, *, model: str, response_format: str = "pcm", sample_rate: int = 24000) -> TTS:
    client = openai.AsyncClient(
        api_key="test",
        base_url="https://compatible.example.com/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return TTS(model=model, client=client, response_format=response_format, sample_rate=sample_rate)


async def _synthesize(tts: TTS) -> bytes:
    audio = b""
    async with tts.synthesize("hello") as stream:
        async for ev in stream:
            audio += ev.frame.data.tobytes()
    return audio


@pytest.mark.parametrize("sample_rate", [-1, 0, 9, 10.5, float("nan"), float("inf")])
def test_tts_rejects_invalid_sample_rate(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate must be an integer of at least 10 Hz"):
        TTS(api_key="test", sample_rate=sample_rate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sample_rate", "expected_rate"), [(None, 24000), (10, 10), (44100, 44100)]
)
async def test_tts_preserves_pcm_and_frame_metadata_for_sample_rate(
    sample_rate: int | None, expected_rate: int
) -> None:
    samples = bytes(i % 256 for i in range((expected_rate // 5 + 1) * 2))

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=samples, headers={"content-type": "audio/pcm"})

    kwargs = {} if sample_rate is None else {"sample_rate": sample_rate}
    tts = _tts(handler, model="tts-1", **kwargs)
    events = []
    async with tts.synthesize("hello") as stream:
        async for event in stream:
            events.append(event)

    assert tts.sample_rate == expected_rate
    frames = [event.frame for event in events]
    assert frames
    assert all(frame.sample_rate == expected_rate for frame in frames)
    assert all(frame.num_channels == 1 for frame in frames)
    audio = b"".join(frame.data.tobytes() for frame in frames)
    assert audio[: len(samples)] == samples
    assert not any(audio[len(samples) :])
    assert not any(event.is_final for event in events[:-1])
    assert events[-1].is_final


@pytest.mark.parametrize(
    ("model", "expected_format"),
    [("tts-1", "audio"), ("tts-1-hd", "audio"), ("gpt-4o-mini-tts", "sse")],
)
async def test_tts_plain_audio_body_is_decoded_whatever_format_was_requested(
    model: str, expected_format: str
) -> None:
    """A compatible server may ignore stream_format and answer with plain audio bytes."""
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, content=PCM, headers={"content-type": "audio/pcm"})

    audio = await _synthesize(_tts(handler, model=model))

    # sse is not supported for tts-1/tts-1-hd, so those must keep requesting audio
    assert requests[0]["stream_format"] == expected_format
    assert audio[: len(PCM)] == PCM


async def test_tts_declared_content_type_wins_over_requested_format() -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(PCM)
    wav = buf.getvalue()

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=wav, headers={"content-type": "audio/wav"})

    # mp3 is requested, but the server answers with wav and says so
    audio = await _synthesize(_tts(handler, model="kokoro", response_format="mp3"))

    assert audio


async def test_tts_sse_body_is_parsed() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        events = [
            json.dumps({"type": "speech.audio.delta", "delta": base64.b64encode(PCM).decode()}),
            json.dumps({"type": "speech.audio.done", "usage": {"input_tokens": 1}}),
        ]
        body = "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    audio = await _synthesize(_tts(handler, model="gpt-4o-mini-tts"))

    assert audio[: len(PCM)] == PCM


async def test_tts_a_second_prewarm_does_not_replace_one_still_in_flight() -> None:
    """aclose() cancels only the task it holds, so a replaced prewarm would outlive the close."""
    release = asyncio.Event()
    in_flight = asyncio.Semaphore(0)

    async def handler(request: httpx2.Request) -> httpx2.Response:
        in_flight.release()
        await release.wait()
        return httpx2.Response(200, content=b"", headers={"content-type": "text/plain"})

    tts = _tts(handler, model="gpt-4o-mini-tts")
    tts.prewarm()
    first = tts._prewarm_task
    await in_flight.acquire()
    tts.prewarm()

    await tts.aclose()
    try:
        assert first is not None and first.done(), "a prewarm task outlived aclose()"
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
