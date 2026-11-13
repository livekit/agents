"""OpenAI STT and TTS against a local OpenAI-compatible peer, and how their failures surface."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ParamSpec, TypedDict

import openai
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import MultiDictProxy

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    TokenUsage,
    http,
    llm,
    stt,
    tts,
)
from livekit.plugins.openai import TTS, responses, stt as openai_stt
from tests.fakes import FakeVAD, audio_frame

NO_RETRY = APIConnectOptions(max_retry=0)
PCM = b"\x00\x01" * 4800  # 200 ms of 16-bit mono at 24 kHz
P = ParamSpec("P")
Route = tuple[str, str, Callable[[web.Request], Awaitable[web.StreamResponse]]]
Form = MultiDictProxy[str | bytes | bytearray | web.FileField]


class _Transcription(TypedDict, total=False):
    """The transcription settings of a realtime session.update."""

    prompt: str
    keywords: list[str]
    languages: list[str]
    language: str


def _silence() -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\x00\x00" * 2400, 24000, 1, 2400)


@asynccontextmanager
async def _serve(*routes: Route) -> AsyncIterator[str]:
    """A local server for ``(method, path, handler)`` routes; yields its ``/v1`` base url."""
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    async with TestServer(app) as server, http.open():
        yield str(server.make_url("/v1"))


class _RealtimePeer:
    """A realtime transcription endpoint: keeps each socket and the events sent on it."""

    def __init__(self) -> None:
        self.sockets: list[web.WebSocketResponse] = []
        self.queries: list[dict[str, str]] = []
        self.sent: list[dict[str, object]] = []
        self.transcriptions: list[_Transcription] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.queries.append(dict(request.query))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets.append(socket)
        async for message in socket:
            event = json.loads(message.data)
            self.sent.append(event)
            if event["type"] == "session.update":
                self.transcriptions.append(event["session"]["audio"]["input"]["transcription"])
        return socket

    async def configured(self) -> _Transcription:
        """The transcription settings of the first session.update."""
        async with asyncio.timeout(3):
            while not self.transcriptions:
                await asyncio.sleep(0.01)
        return self.transcriptions[0]


async def _until_final(stream: stt.RecognizeStream) -> stt.SpeechEvent:
    async with asyncio.timeout(3):
        async for event in stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                return event
    raise AssertionError("the stream ended without a final transcript")


# STT


def _kept(
    make: Callable[P, openai.AsyncClient], made: list[openai.AsyncClient]
) -> Callable[P, openai.AsyncClient]:
    """``make``, keeping every client it makes in ``made``."""

    def make_and_keep(*args: P.args, **kwargs: P.kwargs) -> openai.AsyncClient:
        made.append(make(*args, **kwargs))
        return made[-1]

    return make_and_keep


async def test_realtime_transcription_is_configured_with_keywords_and_languages() -> None:
    """The realtime session carries the prompt, keywords and languages, never `language` too.

    https://github.com/livekit/agents/pull/6705
    """
    peer = _RealtimePeer()
    async with _serve(("GET", "/v1/realtime", peer.handle)) as base_url:
        model = openai_stt.STT(
            api_key="test",
            base_url=base_url,
            model="gpt-live-transcribe",
            prompt="A customer support call.",
            keywords=["premium plan", "AC-42"],
            language=["en", "fr"],
            vad=None,
        )
        stream = model.stream(conn_options=NO_RETRY)
        try:
            transcription = await peer.configured()
        finally:
            await stream.aclose()
            await model.aclose()

    assert transcription["prompt"] == "A customer support call."
    assert transcription["keywords"] == ["premium plan", "AC-42"]
    assert transcription["languages"] == ["en", "fr"]
    assert "language" not in transcription
    assert peer.queries == [{"intent": "transcription", "model": "gpt-live-transcribe"}]


async def test_file_transcription_sends_list_fields_in_bracketed_form() -> None:
    """Keywords and languages go out as repeated `name[]` form fields.

    https://github.com/livekit/agents/pull/6705
    """
    forms: list[Form] = []

    async def transcribe(request: web.Request) -> web.Response:
        forms.append(await request.post())
        return web.json_response({"text": "hello"})

    async with _serve(("POST", "/v1/audio/transcriptions", transcribe)) as base_url:
        model = openai_stt.STT(
            api_key="test",
            base_url=base_url,
            model="gpt-transcribe",
            keywords=["premium plan", "AC-42"],
            language=["en", "fr"],
        )
        try:
            event = await model.recognize(_silence(), conn_options=NO_RETRY)
        finally:
            await model.aclose()

    assert event.alternatives[0].text == "hello"
    [form] = forms
    assert form.getall("keywords[]") == ["premium plan", "AC-42"]
    assert form.getall("languages[]") == ["en", "fr"]
    assert "language" not in form


@pytest.mark.parametrize("empty_code", [None, ""])
async def test_realtime_reports_the_detected_language(empty_code: str | None) -> None:
    """A final transcript names the first language the service detected.

    https://github.com/livekit/agents/pull/6705
    """
    peer = _RealtimePeer()
    async with _serve(("GET", "/v1/realtime", peer.handle)) as base_url:
        model = openai_stt.STT(
            api_key="test",
            base_url=base_url,
            model="gpt-live-transcribe",
            language=["en", "fr"],
            vad=None,
        )
        stream = model.stream(conn_options=NO_RETRY)
        try:
            await peer.configured()
            await peer.sockets[0].send_json(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_1",
                    "transcript": "bonjour, hello",
                    "languages": [{"code": empty_code}, {"code": "fr"}, {"code": "en"}],
                }
            )
            final = await _until_final(stream)
        finally:
            await stream.aclose()
            await model.aclose()

    assert (final.alternatives[0].text, final.alternatives[0].language) == ("bonjour, hello", "fr")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/pull/7828 reproduces: per-call language leaks into "
    "later transcription requests",
)
async def test_a_language_override_applies_only_to_its_request() -> None:
    """A per-call language does not stick to the next transcription.

    https://github.com/livekit/agents/pull/7828
    """
    forms: list[Form] = []

    async def transcribe(request: web.Request) -> web.Response:
        forms.append(await request.post())
        return web.json_response({"text": "hello"})

    async with _serve(("POST", "/v1/audio/transcriptions", transcribe)) as base_url:
        model = openai_stt.STT(api_key="test", base_url=base_url, language="en")
        try:
            first = await model.recognize(_silence(), language="fr")
            second = await model.recognize(_silence())
        finally:
            await model.aclose()

    assert [form["language"] for form in forms] == ["fr", "en"]
    assert [first.alternatives[0].language, second.alternatives[0].language] == ["fr", "en"]


UTTERANCES = [
    {"type": "input_audio_buffer.speech_started", "item_id": "one", "audio_start_ms": 0},
    {"type": "conversation.item.input_audio_transcription.delta", "item_id": "one", "delta": "hel"},
    {"type": "conversation.item.input_audio_transcription.delta", "item_id": "one", "delta": "lo"},
    {"type": "input_audio_buffer.speech_stopped", "item_id": "one", "audio_end_ms": 600},
    {"type": "input_audio_buffer.speech_started", "item_id": "two", "audio_start_ms": 800},
    {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "one",
        "transcript": "hello",
    },
    {"type": "input_audio_buffer.speech_stopped", "item_id": "two", "audio_end_ms": 1100},
    {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "two",
        "transcript": None,
    },
    {
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "three",
        "delta": "bye",
    },
    {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "three",
        "transcript": "bye",
        "usage": {"type": "duration", "seconds": 0.4},
    },
]


@pytest.mark.parametrize("local_vad", [False, True], ids=["server_vad", "local_vad"])
async def test_realtime_keeps_turn_boundaries_and_late_final_transcripts(local_vad: bool) -> None:
    """Speech boundaries come from one source, and a final arriving after the next turn began
    still lands on its own item.

    https://github.com/livekit/agents/issues/3312 https://github.com/livekit/agents/issues/3331
    https://github.com/livekit/agents/pull/4232 https://github.com/livekit/agents/pull/6297
    https://github.com/livekit/agents/pull/2423
    """
    peer = _RealtimePeer()
    async with _serve(("GET", "/v1/realtime", peer.handle)) as base_url:
        model = openai_stt.STT(
            api_key="test",
            base_url=base_url,
            use_realtime=True,
            detect_language=True,
            vad=FakeVAD() if local_vad else None,
        )
        stream = model.stream(conn_options=NO_RETRY)
        events: list[stt.SpeechEvent] = []
        try:
            transcription = await peer.configured()
            if local_vad:
                stream.push_frame(audio_frame(0.2))
                stream.push_frame(audio_frame(0.3, level=0))
                async with asyncio.timeout(3):
                    while not events or events[-1].type != stt.SpeechEventType.END_OF_SPEECH:
                        events.append(await anext(stream))
            for frame in UTTERANCES:
                await peer.sockets[0].send_json(frame)
            async with asyncio.timeout(3):
                async for event in stream:
                    events.append(event)
                    if sum(e.type == stt.SpeechEventType.RECOGNITION_USAGE for e in events) == 3:
                        break
        finally:
            await stream.aclose()
            await model.aclose()

    assert "language" not in transcription
    boundaries = [
        e.type
        for e in events
        if e.type in (stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH)
    ]
    turns = 1 if local_vad else 2
    assert (
        boundaries
        == [stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH] * turns
    )
    assert not any(e["type"] == "input_audio_buffer.commit" for e in peer.sent)
    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [(e.request_id, e.alternatives[0].text) for e in finals] == [
        ("one", "hello"),
        ("three", "bye"),
    ]
    usage = [e.recognition_usage for e in events if e.recognition_usage is not None]
    assert [u.audio_duration for u in usage] == pytest.approx([0.6, 0.3, 0.4])


async def test_closing_the_model_closes_its_pooled_realtime_socket() -> None:
    """A finished stream returns its socket to the pool; closing the model closes it.

    https://github.com/livekit/agents/pull/7023
    """
    peer = _RealtimePeer()
    async with _serve(("GET", "/v1/realtime", peer.handle)) as base_url:
        model = openai_stt.STT(api_key="test", base_url=base_url, use_realtime=True, vad=None)
        stream = model.stream(conn_options=NO_RETRY)
        await peer.configured()
        await stream.aclose()
        assert not peer.sockets[0].closed

        await model.aclose()

        async with asyncio.timeout(3):
            while not peer.sockets[0].closed:
                await asyncio.sleep(0.01)


# TTS


def _speaking(
    body: bytes, content_type: str, requests: list[dict[str, object]] | None = None
) -> Route:
    async def speech(request: web.Request) -> web.Response:
        if requests is not None:
            requests.append(await request.json())
        return web.Response(body=body, headers={"content-type": content_type})

    return ("POST", "/v1/audio/speech", speech)


async def _synthesize(base_url: str, model: str = "gpt-4o-mini-tts") -> list[tts.SynthesizedAudio]:
    speech = TTS(api_key="test", base_url=base_url, model=model, response_format="pcm")
    try:
        async with speech.synthesize("hello", conn_options=NO_RETRY) as stream:
            return [event async for event in stream]
    finally:
        await speech.aclose()


@pytest.mark.parametrize("model", ["hexgrad/Kokoro-82M", "gpt-4o-mini-tts"])
async def test_a_plain_audio_body_is_decoded_whatever_the_model(model: str) -> None:
    """A compatible server that ignores stream_format and answers raw audio is still heard.

    https://github.com/livekit/agents/pull/6930
    """
    async with _serve(_speaking(PCM, "audio/pcm")) as base_url:
        frames = await _synthesize(base_url, model)

    assert b"".join(f.frame.data.tobytes() for f in frames)[: len(PCM)] == PCM


async def test_an_event_stream_body_is_decoded_with_its_usage() -> None:
    """OpenAI's SSE answer is decoded, its usage carried on the final frame.

    https://github.com/livekit/agents/pull/6930
    """
    usage = {
        "input_tokens": 14,
        "output_tokens": 101,
        "total_tokens": 115,
        "input_token_details": {"text_tokens": 14, "audio_tokens": 0},
    }
    events = [
        {"type": "speech.audio.delta", "delta": base64.b64encode(PCM).decode()},
        {"type": "speech.audio.done", "usage": usage},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    async with _serve(_speaking(body.encode(), "text/event-stream")) as base_url:
        frames = await _synthesize(base_url)

    assert b"".join(f.frame.data.tobytes() for f in frames)[: len(PCM)] == PCM
    assert frames[-1].is_final
    assert frames[-1].usage == TokenUsage(
        input_tokens=14, output_tokens=101, input_text_tokens=14, input_audio_tokens=0
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [("tts-1", "audio"), ("tts-1-hd", "audio"), ("gpt-4o-mini-tts", "sse")],
)
async def test_stream_format_is_requested_per_model(model: str, expected: str) -> None:
    """tts-1 and tts-1-hd do not support SSE, so they keep asking for raw audio.

    https://github.com/livekit/agents/pull/6930
    """
    requests: list[dict[str, object]] = []
    async with _serve(_speaking(PCM, "audio/pcm", requests)) as base_url:
        await _synthesize(base_url, model)

    assert requests[0]["stream_format"] == expected


async def test_a_second_prewarm_does_not_open_another_request() -> None:
    """A prewarm still in flight is kept, so closing the TTS settles the only one there is.

    https://github.com/livekit/agents/pull/7147
    """
    requests: list[str] = []
    release = asyncio.Event()

    async def warm(request: web.Request) -> web.Response:
        requests.append(request.path)
        await release.wait()
        return web.Response(text="")

    async with _serve(("GET", "/v1/", warm)) as base_url:
        tts = TTS(api_key="test", base_url=base_url)
        tts.prewarm()
        async with asyncio.timeout(3):
            while not requests:
                await asyncio.sleep(0.01)
        tts.prewarm()
        await asyncio.sleep(0.05)
        await tts.aclose()
        release.set()

    assert requests == ["/v1/"]


@pytest.mark.parametrize("chunk_size", [1, 137, 8192])
async def test_opus_split_across_http_chunks_decodes_completely(chunk_size: int) -> None:
    """Opus arriving in arbitrary chunks decodes to the same audio as one body.

    https://github.com/livekit/agents/pull/1494
    """
    opus = (Path(__file__).parents[1] / "fixtures" / "change-sophie.opus").read_bytes()
    sizes = iter([len(opus), chunk_size])

    async def speech(request: web.Request) -> web.StreamResponse:
        size = next(sizes)
        response = web.StreamResponse(headers={"content-type": "audio/ogg"})
        await response.prepare(request)
        for offset in range(0, len(opus), size):
            await response.write(opus[offset : offset + size])
        return response

    async with _serve(("POST", "/v1/audio/speech", speech)) as base_url:
        tts = TTS(api_key="test", base_url=base_url, model="tts-1", response_format="opus")
        heard = []
        try:
            for _ in range(2):
                async with tts.synthesize("change Sophie", conn_options=NO_RETRY) as stream:
                    heard.append(b"".join([e.frame.data.tobytes() async for e in stream]))
        finally:
            await tts.aclose()

    assert heard[0] and heard[0] == heard[1]


# failures


def _chat_ctx() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hi")
    return ctx


async def _call(provider: str, base_url: str, options: APIConnectOptions) -> None:
    if provider == "stt":
        recognizer = openai_stt.STT(api_key="test", base_url=base_url)
        try:
            await recognizer.recognize(_silence(), conn_options=options)
        finally:
            await recognizer.aclose()
    elif provider == "tts":
        speech = TTS(api_key="test", base_url=base_url, response_format="pcm")
        try:
            async with speech.synthesize("hello", conn_options=options) as audio:
                _ = [event async for event in audio]
        finally:
            await speech.aclose()
    else:
        model = responses.LLM(api_key="test", base_url=base_url, use_websocket=False)
        try:
            async with model.chat(chat_ctx=_chat_ctx(), conn_options=options) as reply:
                _ = [chunk async for chunk in reply]
        finally:
            await model.aclose()


@pytest.mark.parametrize("provider", ["stt", "tts", "responses"])
@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("timeout", APITimeoutError),
        ("connection", APIConnectionError),
        ("status", APIStatusError),
        ("fatal_status", APIStatusError),
    ],
)
async def test_a_failure_keeps_its_category_and_is_not_retried_by_the_sdk(
    provider: str, failure: str, error: type[APIError]
) -> None:
    """A failed request raises its own error type with the server's details, once, leaving
    retries to the framework.

    https://github.com/livekit/agents/pull/6341 https://github.com/livekit/agents/pull/2020
    https://github.com/livekit/agents/pull/1674
    """
    requests: list[str] = []

    async def fail(request: web.Request) -> web.StreamResponse:
        requests.append(request.headers["authorization"])
        if failure == "timeout":
            await asyncio.Event().wait()  # never answers
        if failure == "connection":
            assert request.transport is not None
            request.transport.close()
        return web.json_response(
            {"error": {"message": "quota exhausted", "type": "rate_limit_error"}},
            status=400 if failure == "fatal_status" else 429,
            headers={"x-request-id": "request-quota"},
        )

    routes = [("POST", path, fail) for path in ("/v1/audio/transcriptions", "/v1/audio/speech")]
    # a silent server times the read out at once
    options = APIConnectOptions(max_retry=0, read_timeout=0.05 if failure == "timeout" else 5)
    async with _serve(*routes, ("POST", "/v1/responses", fail)) as base_url:
        with pytest.raises(error) as caught:
            await _call(provider, base_url, options)

    assert requests == ["Bearer test"]
    assert caught.value.retryable == (failure != "fatal_status")
    if isinstance(caught.value, APIStatusError):
        assert caught.value.status_code == (400 if failure == "fatal_status" else 429)
        assert caught.value.request_id == "request-quota"
        assert "quota exhausted" in str(caught.value)


async def test_closing_the_tts_closes_the_client_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TTS closes the SDK client it made.

    https://github.com/livekit/agents/pull/4735
    """
    created: list[openai.AsyncClient] = []
    make_client = openai.AsyncClient

    monkeypatch.setattr(openai, "AsyncClient", _kept(make_client, created))
    tts = TTS(api_key="test")

    await tts.aclose()

    assert [client.is_closed() for client in created] == [True]
