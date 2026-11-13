"""ElevenLabs Scribe STT against a local peer speaking the Scribe JSON protocol."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypedDict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from typing_extensions import Unpack
from yarl import URL

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, http, stt
from livekit.plugins.elevenlabs import STT, STTRealtimeSampleRates
from livekit.plugins.elevenlabs.stt import SpeechStream, VADOptions

OPTIONS = APIConnectOptions(max_retry=0, read_timeout=0.5)
SERVER_VAD: VADOptions = {"vad_silence_threshold_secs": 0.5}
JSON = dict[str, object]


class ScribeOptions(TypedDict, total=False):
    server_vad: VADOptions
    language_code: str
    include_language_detection: bool
    secondary_languages: list[str]
    include_timestamps: bool
    keyterms: list[str]


class HandshakeOptions(ScribeOptions, total=False):
    model: str


def silence(ms: int, rate: int = 16000) -> rtc.AudioFrame:
    samples = rate * ms // 1000
    return rtc.AudioFrame(
        data=bytes(samples * 2), sample_rate=rate, num_channels=1, samples_per_channel=samples
    )


@asynccontextmanager
async def peer(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[tuple[str, http.HttpSession]]:
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    async with TestServer(app) as server, http.open() as session:
        yield str(server.make_url("")).rstrip("/"), session


async def transcribe(
    replies: list[JSON], *, until: stt.SpeechEventType, **options: Unpack[ScribeOptions]
) -> list[stt.SpeechEvent]:
    """Stream audio to a Scribe peer that answers it with `replies`; events up to `until`."""

    async def scribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        for reply in replies:
            await ws.send_json(reply)
        async for _ in ws:
            pass
        return ws

    events: list[stt.SpeechEvent] = []
    options = {"include_language_detection": False, **options}
    async with peer(scribe) as (url, session):
        provider = STT(
            api_key="key",
            model="scribe_v2_realtime",
            base_url=url,
            http_session=session,
            **options,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(silence(100))
            async with asyncio.timeout(2):
                async for event in stream:
                    events.append(event)
                    if event.type == until:
                        break
        await provider.aclose()
    return [e for e in events if e.type != stt.SpeechEventType.RECOGNITION_USAGE]


def committed(text: str, **extra: object) -> JSON:
    words = [{"text": text, "start": 0.1, "end": 0.4}] if text else []
    return {"message_type": "committed_transcript", "text": text, "words": words, **extra}


async def handshake_path(**options: Unpack[HandshakeOptions]) -> str:
    """The raw request path of the Scribe websocket handshake for these options."""
    paths: list[str] = []
    connected = asyncio.Event()

    async def scribe(request: web.Request) -> web.WebSocketResponse:
        paths.append(request.raw_path)
        connected.set()
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    async with peer(scribe) as (url, session):
        provider = STT(api_key="key", base_url=url, http_session=session, **options)
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(silence(100))
            await asyncio.wait_for(connected.wait(), 2)
        await provider.aclose()
    return paths[0]


async def test_re_sent_partial_transcript_is_dropped() -> None:
    """A partial the server repeats is emitted once. https://github.com/livekit/agents/pull/6922"""
    partial: JSON = {"message_type": "partial_transcript", "text": "yeah please", "words": []}

    events = await transcribe(
        [partial] * 5 + [committed("yeah please")],
        until=stt.SpeechEventType.END_OF_SPEECH,
        server_vad=SERVER_VAD,
    )

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_server_vad_commit_ends_the_turn() -> None:
    """With server VAD a commit is final and ends speech. https://github.com/livekit/agents/issues/5849"""
    events = await transcribe(
        [committed("hello")], until=stt.SpeechEventType.END_OF_SPEECH, server_vad=SERVER_VAD
    )

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "hello"


async def test_manual_commit_ends_speech_only_on_the_empty_commit() -> None:
    """Without server VAD, speech ends on the empty commit. https://github.com/livekit/agents/pull/5872"""
    events = await transcribe(
        [committed("hello"), committed("again"), committed("")],
        until=stt.SpeechEventType.END_OF_SPEECH,
    )

    assert [e.type for e in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_committed_transcript_carries_word_confidence() -> None:
    """Word log-probabilities become the final's confidence. https://github.com/livekit/agents/pull/6461"""
    word = {"text": "hello", "start": 0.1, "end": 0.4, "type": "word", "logprob": -0.02}

    events = await transcribe(
        [{"message_type": "committed_transcript", "text": "hello", "words": [word]}],
        until=stt.SpeechEventType.FINAL_TRANSCRIPT,
        server_vad=SERVER_VAD,
    )

    assert events[-1].alternatives[0].confidence > 0.9


async def test_final_transcript_reports_the_detected_language() -> None:
    """With detection on, the final carries the language the model heard.

    https://github.com/livekit/agents/pull/6902
    """
    events = await transcribe(
        [
            committed("привет"),
            committed(
                "привет",
                message_type="committed_transcript_with_timestamps",
                language_code="ru",
            ),
        ],
        until=stt.SpeechEventType.END_OF_SPEECH,
        server_vad=SERVER_VAD,
        language_code="en",
        include_language_detection=True,
        secondary_languages=["ru"],
    )

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [f.alternatives[0].language for f in finals] == ["ru"]


@pytest.mark.parametrize(
    "message_type",
    ["partial_transcript", "committed_transcript", "committed_transcript_with_timestamps"],
)
@pytest.mark.parametrize(
    ("language", "language_data", "expected_language"),
    [
        ("es", {}, "es"),
        ("es", {"language_code": None}, "es"),
        ("es", {"language_code": ""}, "es"),
        ("es", {"language_code": "fra"}, "fr"),
        (None, {}, "en"),
        (None, {"language_code": None}, "en"),
        (None, {"language_code": ""}, "en"),
        (None, {"language_code": "fra"}, "fr"),
    ],
)
async def test_transcript_language_falls_back_to_configured_language(
    message_type: str,
    language: str | None,
    language_data: dict[str, str | None],
    expected_language: str,
) -> None:
    """A transcript without a language reports the configured one, else English.

    https://github.com/livekit/agents/pull/7306
    """
    partial = message_type == "partial_transcript"
    options: ScribeOptions = {
        "include_timestamps": message_type == "committed_transcript_with_timestamps"
    }
    if language:
        options["language_code"] = language

    events = await transcribe(
        [{"message_type": message_type, "text": "hola", **language_data}],
        until=stt.SpeechEventType.INTERIM_TRANSCRIPT
        if partial
        else stt.SpeechEventType.FINAL_TRANSCRIPT,
        **options,
    )

    assert events[-1].alternatives[0].language == expected_language


async def test_keyterms_with_query_delimiters_survive_the_url() -> None:
    """Keyterms containing `&` or `#` reach the server intact. https://github.com/livekit/agents/pull/6746"""
    path = await handshake_path(model="scribe_v2_realtime", keyterms=["Smith & Sons", "C#"])

    assert URL(path).query.getall("keyterms") == ["Smith & Sons", "C#"]


async def test_languages_go_on_the_wire_without_their_region() -> None:
    """The realtime API rejects "en-US", so languages are sent bare. https://github.com/livekit/agents/pull/6902"""
    path = await handshake_path(
        model="scribe_v2_realtime", language_code="en_US", secondary_languages=["ru-RU"]
    )

    assert URL(path).query.getall("language_code") == ["en"]
    assert URL(path).query.getall("secondary_languages") == ["ru"]


async def stream_packets(push: Callable[[SpeechStream], None], count: int) -> list[JSON]:
    """The first `count` packets a Scribe peer receives while `push` feeds the stream."""
    packets: list[JSON] = []
    received = asyncio.Event()

    async def scribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packets.append(json.loads(msg.data))
            if len(packets) == count:
                received.set()
        return ws

    async with peer(scribe) as (url, session):
        provider = STT(
            api_key="key", model="scribe_v2_realtime", base_url=url, http_session=session
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            push(stream)
            await asyncio.wait_for(received.wait(), 2)
        await provider.aclose()
    return packets


async def test_flush_commits_the_turn_when_no_audio_is_left_to_send() -> None:
    """A flush with no buffered audio still commits the turn. https://github.com/livekit/agents/pull/6598"""

    def push(stream: SpeechStream) -> None:
        stream.push_frame(silence(50))  # exactly one 50 ms chunk, nothing left to flush
        stream.flush()

    packets = await stream_packets(push, 2)

    assert [(p["commit"], p["audio_base_64"] == "") for p in packets] == [
        (False, False),
        (True, True),
    ]


async def test_flush_commits_after_the_buffered_audio() -> None:
    """The tail of a turn reaches the server before its commit. https://github.com/livekit/agents/pull/6598"""

    def push(stream: SpeechStream) -> None:
        stream.push_frame(silence(30))
        stream.flush()

    packets = await stream_packets(push, 2)

    assert [(p["commit"], p["audio_base_64"] == "") for p in packets] == [
        (False, False),
        (True, True),
    ]


_INVALID_KEY = {"detail": {"code": "invalid_api_key", "message": "Invalid API key"}}


@pytest.mark.parametrize(
    ("body", "detail", "parsed"),
    [
        (json.dumps(_INVALID_KEY), "Invalid API key", _INVALID_KEY),
        ("", None, ""),
        ("{not json", "{not json", "{not json"),
    ],
)
async def test_recognize_does_not_retry_a_client_error(
    body: str, detail: str | None, parsed: object
) -> None:
    """A 4xx from batch recognition raises once, the provider's message in its detail.

    https://github.com/livekit/agents/pull/7701
    """
    requests = 0

    async def reject(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(status=401, text=body, content_type="application/json")

    async with peer(reject) as (url, session):
        provider = STT(api_key="bad-key", base_url=url, http_session=session)
        with pytest.raises(APIStatusError) as error:
            await provider.recognize([silence(100)])

    assert (error.value.status_code, error.value.detail, error.value.body) == (
        401,
        detail,
        parsed,
    )
    assert requests == 1


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
async def test_batch_recognition_honors_custom_url_and_tagging(
    rate: STTRealtimeSampleRates,
) -> None:
    """Batch recognition posts to the custom URL with the configured options.

    https://github.com/livekit/agents/issues/3226
    """
    forms = []

    async def recognize(request: web.Request) -> web.Response:
        forms.append(await request.post())
        return web.json_response({"text": "Hello", "language_code": "en", "words": []})

    async with peer(recognize) as (url, session):
        provider = STT(
            api_key="key",
            base_url=url,
            http_session=session,
            sample_rate=rate,
            tag_audio_events=False,
        )
        result = await provider.recognize(silence(500, rate), conn_options=OPTIONS)
        await provider.aclose()

    assert result.alternatives[0].text == "Hello"
    assert forms[0]["tag_audio_events"] == "false"


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
async def test_realtime_sends_the_configured_rate(rate: STTRealtimeSampleRates) -> None:
    """Streamed audio keeps its sample rate.

    https://github.com/livekit/agents/issues/5573 https://github.com/livekit/agents/pull/4953
    """
    packets, queries = [], []

    async def scribe(request: web.Request) -> web.WebSocketResponse:
        queries.append(request.query["audio_format"])
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            if packet.get("commit"):
                await ws.send_json({"message_type": "committed_transcript", "text": "Hello"})
        return ws

    async with peer(scribe) as (url, session):
        provider = STT(
            api_key="key",
            model="scribe_v2_realtime",
            include_language_detection=False,
            base_url=url,
            http_session=session,
            sample_rate=rate,
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(silence(500, rate))
            stream.flush()
            async with asyncio.timeout(2):
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        break
        await provider.aclose()

    assert queries == [f"pcm_{rate}"]
    assert {p["sample_rate"] for p in packets} == {rate}
    assert sum(len(base64.b64decode(p["audio_base_64"])) for p in packets) == rate


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7992 reproduces: short Scribe commits are not padded to 300ms",
)
async def test_short_scribe_commit_keeps_connection() -> None:
    """A short commit is padded so Scribe can return its final transcript. https://github.com/livekit/agents/pull/7992"""

    async def scribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        samples = 0
        async for msg in ws:
            packet = json.loads(msg.data)
            samples += len(base64.b64decode(packet["audio_base_64"])) // 2
            if packet.get("commit"):
                if samples < 4800:
                    await ws.close(code=1008, message=b"commit too short")
                else:
                    await ws.send_json({"message_type": "committed_transcript", "text": "Hello"})
        return ws

    finals = []
    async with peer(scribe) as (url, session):
        provider = STT(
            api_key="key",
            model="scribe_v2_realtime",
            include_language_detection=False,
            base_url=url,
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=OPTIONS) as stream:
                stream.push_frame(silence(50))
                stream.flush()
                async with asyncio.timeout(2):
                    async for event in stream:
                        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                            finals.append(event.alternatives[0].text)
                            break
        except APIError:
            pass
        finally:
            await provider.aclose()

    assert finals == ["Hello"]


async def test_scribe_rejected_handshake_is_typed() -> None:
    """A rejected Scribe handshake raises its HTTP status. https://github.com/livekit/agents/issues/4201"""

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401, text="unauthorized")

    async with peer(reject) as (url, session):
        provider = STT(
            api_key="key", base_url=url, model="scribe_v2_realtime", http_session=session
        )
        async with provider.stream(conn_options=OPTIONS) as stream:
            with pytest.raises(APIStatusError) as error:
                async with asyncio.timeout(2):
                    _ = [ev async for ev in stream]
        await provider.aclose()

    assert error.value.status_code == 401
