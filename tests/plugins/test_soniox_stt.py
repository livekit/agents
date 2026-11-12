"""Soniox STT: tokens from a local WebSocket peer become speech events."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIStatusError, LanguageCode, http, stt
from livekit.agents.stt import SpeechEventType
from livekit.plugins.soniox import STT, STTOptions
from livekit.plugins.soniox.stt import TranslationConfig

END = {"text": "<end>", "is_final": True}
TWO_WAY = TranslationConfig(type="two_way", language_a="en", language_b="es")


def token(
    text: str, language: str = "en", *, final: bool = True, **fields: object
) -> dict[str, object]:
    return {"text": text, "language": language, "is_final": final, **fields}


async def recognize(
    messages: Sequence[Mapping[str, object]], count: int, params: STTOptions | None = None
) -> list[stt.SpeechEvent]:
    """The first ``count`` events of a stream whose peer sends ``messages`` after the config."""

    async def soniox(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        for message in messages:
            await ws.send_json(message)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", soniox)
    async with TestServer(app) as server, http.open():
        model = STT(api_key="key", base_url=str(server.make_url("/transcribe")), params=params)
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
        try:
            return [await anext(stream) for _ in range(count)]
        finally:
            await stream.aclose()
            await model.aclose()


def first(events: list[stt.SpeechEvent], *types: SpeechEventType) -> stt.SpeechData:
    return next(e for e in events if e.type in types).alternatives[0]


@pytest.mark.parametrize("level", [None, 0, 2])
async def test_key_goes_in_the_header_and_latency_level_in_the_config(level: int | None) -> None:
    """The API key is a bearer header, never config; the latency level is sent only when set."""
    configs: asyncio.Queue[tuple[str | None, dict[str, object]]] = asyncio.Queue()

    async def soniox(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        configs.put_nowait((request.headers.get("Authorization"), await ws.receive_json()))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", soniox)
    async with TestServer(app) as server, http.open():
        model = STT(
            api_key="test-key",
            base_url=str(server.make_url("/transcribe")),
            params=STTOptions(endpoint_latency_adjustment_level=level),
        )
        stream = model.stream()
        authorization, config = await configs.get()
        await stream.aclose()
        await model.aclose()

    assert authorization == "Bearer test-key"
    assert "api_key" not in config
    assert config.get("endpoint_latency_adjustment_level") == level


async def test_authentication_error_surfaces_without_retry() -> None:
    """A 401 from Soniox fails the stream at once, with no reconnect."""
    connections = 0

    async def soniox(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({"error_code": 401, "error_message": "invalid key"})
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/transcribe", soniox)
    async with TestServer(app) as server, http.open():
        model = STT(api_key="key", base_url=str(server.make_url("/transcribe")))
        stream = model.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))

        with pytest.raises(APIStatusError) as raised:
            await anext(stream)
        await stream.aclose()
        await model.aclose()

    assert raised.value.status_code == 401
    assert not raised.value.retryable
    assert connections == 1


async def test_endpoint_finishes_speech() -> None:
    """An ``<end>`` token closes the utterance. https://github.com/livekit/agents/issues/4034"""
    events = await recognize([{"tokens": [token("hello")]}, {"tokens": [END]}], 4)

    assert [e.type for e in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.PREFLIGHT_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
        SpeechEventType.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "hello"


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7728 reproduces: retracted interim "
    "leaves speech open",
)
async def test_endpoint_after_a_retracted_interim_finishes_speech() -> None:
    """Speech ends even when its only interim was retracted. https://github.com/livekit/agents/issues/7728"""
    messages: list[dict[str, object]] = [
        {"tokens": [token("hello", final=False)]},
        {"tokens": []},
        {"tokens": [END]},
    ]

    async with asyncio.timeout(2):  # the bug is that END_OF_SPEECH never comes
        events = await recognize(messages, 3)

    assert events[-1].type == SpeechEventType.END_OF_SPEECH


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/6887 reproduces: endpoint before "
    "translation discards source text",
)
async def test_endpoint_before_the_translation_keeps_the_source_text() -> None:
    """The source text survives an endpoint that precedes its translation.

    https://github.com/livekit/agents/issues/6887
    """
    params = STTOptions(translation=TranslationConfig(type="one_way", target_language="en"))
    messages = [
        {"tokens": [token("hola", "es", translation_status="original"), END]},
        {"tokens": [token("hello", translation_status="translation"), END]},
    ]

    events = await recognize(messages, 2, params)

    assert events[-1].type == SpeechEventType.END_OF_SPEECH
    assert events[0].alternatives[0].text == "hello"
    assert events[0].alternatives[0].source_texts == ["hola"]


async def test_final_transcript_keeps_speaker_timing_and_confidence() -> None:
    """Final tokens keep diarization, their time span and mean confidence.

    https://github.com/livekit/agents/issues/4509 https://github.com/livekit/agents/pull/4939
    """
    tokens = [
        token("hello ", speaker="speaker-2", start_ms=1250, end_ms=1750, confidence=0.6),
        token("world", speaker="speaker-2", start_ms=2000, end_ms=2750, confidence=0.8),
        END,
    ]

    events = await recognize([{"tokens": tokens}], 2, STTOptions(enable_speaker_diarization=True))

    data = events[0].alternatives[0]
    assert (data.text, data.speaker_id) == ("hello world", "speaker-2")
    assert data.confidence == pytest.approx(0.7)
    assert data.end_time - data.start_time == pytest.approx(1.5)  # shifted by the session offset
    assert data.start_time >= 1.25


async def test_final_span_runs_from_earliest_start_to_latest_end() -> None:
    """Out-of-order token times never shorten the utterance's span.

    https://github.com/livekit/agents/issues/6885
    """
    tokens = [
        token(" He's", start_ms=41000, end_ms=41400),
        token(" trying", start_ms=40000, end_ms=43843),
        token(".", start_ms=43663, end_ms=43700),
        END,
    ]

    events = await recognize([{"tokens": tokens}], 2)

    data = events[0].alternatives[0]
    assert data.end_time - data.start_time == pytest.approx(3.843)


async def test_two_way_translation_reports_per_run_source_and_target() -> None:
    """Code-switched speech keeps each run's language on both sides of the translation.

    https://github.com/livekit/agents/issues/5685
    """
    tokens = [
        token("No hablo español, ", "es", translation_status="original"),
        token("but I speak English.", "en", translation_status="original"),
        token("I don't speak Spanish, ", "en", translation_status="translation"),
        token("pero hablo inglés.", "es", translation_status="translation"),
        END,
    ]

    events = await recognize(
        [{"tokens": tokens, "total_audio_proc_ms": 1000}], 3, STTOptions(translation=TWO_WAY)
    )

    data = first(events, SpeechEventType.FINAL_TRANSCRIPT)
    assert data.text == "I don't speak Spanish, pero hablo inglés."
    assert data.language == LanguageCode("en")
    assert data.source_languages == [LanguageCode("es"), LanguageCode("en")]
    assert data.source_texts == ["No hablo español, ", "but I speak English."]
    assert data.target_languages == [LanguageCode("en"), LanguageCode("es")]
    assert data.target_texts == ["I don't speak Spanish, ", "pero hablo inglés."]


async def test_interim_translation_merges_final_and_pending_tokens_per_run() -> None:
    """https://github.com/livekit/agents/pull/5730"""
    tokens = [
        token("Hola, ", "es", translation_status="original"),
        token("Hello, ", "en", translation_status="translation"),
        token("¿cómo estás?", "es", final=False, translation_status="original"),
        token("how are you?", "en", final=False, translation_status="translation"),
    ]

    events = await recognize([{"tokens": tokens}], 2, STTOptions(translation=TWO_WAY))

    data = first(events, SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.PREFLIGHT_TRANSCRIPT)
    assert (data.source_languages, data.source_texts) == (
        [LanguageCode("es")],
        ["Hola, ¿cómo estás?"],
    )
    assert (data.target_languages, data.target_texts) == (
        [LanguageCode("en")],
        ["Hello, how are you?"],
    )


async def test_code_switched_speech_without_translation_reports_source_runs() -> None:
    """https://github.com/livekit/agents/pull/5730"""
    interim = [token("こんにちは、", "ja"), token("My name is Sam.", "en", final=False)]
    final = [token("こんにちは、君の名前は何だ。", "ja"), token(" My name is Sam.", "en"), END]

    interim_events = await recognize([{"tokens": interim, "total_audio_proc_ms": 600}], 2)
    final_events = await recognize([{"tokens": final, "total_audio_proc_ms": 1500}], 3)

    pending = first(
        interim_events, SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.PREFLIGHT_TRANSCRIPT
    )
    assert pending.source_languages == [LanguageCode("ja"), LanguageCode("en")]
    assert pending.source_texts == ["こんにちは、", "My name is Sam."]
    assert pending.target_texts is None
    done = first(final_events, SpeechEventType.FINAL_TRANSCRIPT)
    assert done.text == "こんにちは、君の名前は何だ。 My name is Sam."
    assert done.source_texts == ["こんにちは、君の名前は何だ。", " My name is Sam."]
    assert done.target_texts is None


async def test_usage_is_reported_for_frames_without_tokens() -> None:
    """Silent audio Soniox processed is still billed. https://github.com/livekit/agents/issues/6584"""
    events = await recognize([{"tokens": [], "total_audio_proc_ms": 3000}], 1)

    assert [e.type for e in events] == [SpeechEventType.RECOGNITION_USAGE]
    assert events[0].recognition_usage is not None
    assert events[0].recognition_usage.audio_duration == pytest.approx(3.0)


async def test_usage_follows_interim_only_frames() -> None:
    """Usage is reported after an interim-only frame's transcript events.

    https://github.com/livekit/agents/issues/6584
    """
    messages = [{"tokens": [token("hola", "es", final=False)], "total_audio_proc_ms": 2500}]

    events = await recognize(messages, 3)

    assert [e.type for e in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.RECOGNITION_USAGE,
    ]
    assert events[-1].recognition_usage is not None
    assert events[-1].recognition_usage.audio_duration == pytest.approx(2.5)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/6435 reproduces: Soniox reports no output "
    "token usage",
)
async def test_usage_counts_only_final_text_tokens_incrementally() -> None:
    """Usage counts final source and translation tokens once. https://github.com/livekit/agents/pull/6435"""
    params = STTOptions(translation=TranslationConfig(type="one_way", target_language="en"))
    messages = [
        {
            "tokens": [token("hola", final=False, translation_status="original")],
            "total_audio_proc_ms": 100,
        },
        {
            "tokens": [
                token("hola", translation_status="original"),
                token("hello", translation_status="translation"),
                END,
            ],
            "total_audio_proc_ms": 200,
        },
        {
            "tokens": [token("bye", translation_status="translation"), token("<fin>")],
            "total_audio_proc_ms": 300,
        },
    ]

    events = await recognize(messages, 7, params)

    usages = [e.recognition_usage for e in events if e.recognition_usage is not None]
    assert [u.tokens.output_tokens for u in usages] == [0, 2, 1]
    assert sum(u.audio_duration or 0.0 for u in usages) == pytest.approx(0.3)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8471 reproduces: per-stream language is ignored",
)
async def test_stream_language_override_stays_local_to_its_stream() -> None:
    """``stream(language=)`` hints only that stream. https://github.com/livekit/agents/pull/8471"""
    configs: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def soniox(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        configs.put_nowait(await ws.receive_json())
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/stt", soniox)
    async with TestServer(app) as server, http.open():
        model = STT(
            api_key="key",
            params=STTOptions(language_hints=["en"]),
            base_url=str(server.make_url("/stt")),
        )
        async with model.stream(language="es"):
            overridden = await configs.get()
        async with model.stream():
            default = await configs.get()
        await model.aclose()

    assert overridden["language_hints"] == ["es"]
    assert default["language_hints"] == ["en"]


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/8311 reproduces: cancellation during "
    "initial send leaks acquired websocket",
)
async def test_closing_during_configuration_closes_the_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream closed while sending its config closes the socket it opened.

    https://github.com/livekit/agents/issues/8311
    """

    async def soniox(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    sockets: list[http.WebSocket] = []
    configuring = asyncio.Event()
    connect = http.HttpSession.ws_connect

    class StalledConfigSend:
        """The opened socket, whose first send (the config) never completes."""

        def __init__(self, socket: http.WebSocket) -> None:
            self.socket = socket

        async def send_str(self, data: str) -> None:
            configuring.set()
            await asyncio.Event().wait()

        async def close(self) -> None:
            await self.socket.close()

    async def stalled_connect(
        self: http.HttpSession,
        url: str,
        *,
        conn_options: APIConnectOptions,
        headers: Mapping[str, str],
        heartbeat: float | None,
    ) -> StalledConfigSend:
        sockets.append(
            await connect(
                self, url, conn_options=conn_options, headers=headers, heartbeat=heartbeat
            )
        )
        return StalledConfigSend(sockets[-1])

    monkeypatch.setattr(http.HttpSession, "ws_connect", stalled_connect)
    app = web.Application()
    app.router.add_get("/stt", soniox)
    async with TestServer(app) as server, http.open():
        model = STT(api_key="key", base_url=str(server.make_url("/stt")))
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
        await configuring.wait()

        await stream.aclose()
        leaked = not sockets[0].closed
        await sockets[0].close()
        await model.aclose()

    assert not leaked
