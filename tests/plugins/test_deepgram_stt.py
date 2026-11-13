"""Deepgram STT (v1 and Flux) against a local peer speaking Deepgram's live and batch wire."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import MultiMapping

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError, http, stt
from livekit.plugins import deepgram
from tests.fakes import until

NO_RETRY = APIConnectOptions(max_retry=0)
AUDIO: dict[str, object] = {"type": "audio"}


def _audio(ms: int) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=16000, num_channels=1, samples_per_channel=16 * ms)


def _server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]], method: str = "GET"
) -> TestServer:
    app = web.Application()
    app.router.add_route(method, "/listen", handler)
    return TestServer(app)


def _result(**channel: object) -> dict[str, object]:
    return {
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "metadata": {"request_id": "regression"},
        "channel": {
            "alternatives": [
                {
                    "transcript": "Hello, world!",
                    "confidence": 0.8,
                    "languages": [],
                    "words": [
                        {
                            "word": "hello",
                            "punctuated_word": "Hello,",
                            "start": 1.25,
                            "end": 1.75,
                            "confidence": 0.7,
                        },
                        {
                            "word": "world",
                            "punctuated_word": "world!",
                            "start": 2.0,
                            "end": 2.75,
                            "confidence": 0.9,
                        },
                    ],
                }
            ],
            **channel,
        },
    }


class _Peer:
    """Accepts every socket and queues what it receives (audio as AUDIO), KeepAlives aside."""

    def __init__(self, *, reply_to_finalize: bool = False) -> None:
        self.reply_to_finalize = reply_to_finalize
        self.sockets: asyncio.Queue[web.WebSocketResponse] = asyncio.Queue()
        self.received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.queries: asyncio.Queue[MultiMapping[str]] = asyncio.Queue()

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.queries.put_nowait(request.query)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.put_nowait(ws)
        async for message in ws:
            if message.type == web.WSMsgType.BINARY:
                self.received.put_nowait(AUDIO)
                continue
            data = message.json()
            if data["type"] != "KeepAlive":
                self.received.put_nowait(data)
            if data["type"] == "Finalize" and self.reply_to_finalize:
                await ws.send_json(_result())
        return ws


def test_update_options_falls_back_to_the_general_model_for_another_language() -> None:
    """An English-only Nova-2 variant becomes nova-2-general for a non-English language.

    https://github.com/livekit/agents/issues/6047 https://github.com/livekit/agents/pull/6041
    """
    model = deepgram.STT(api_key="test", language="fr")

    model.update_options(model="nova-2-meeting")

    assert model.model == "nova-2-general"


async def test_flux_live_fields_reconfigure_in_band_without_reconnecting() -> None:
    """Thresholds, keyterms and language hints go out as one Configure on the open socket.

    https://github.com/livekit/agents/pull/6277
    """
    peer = _Peer()
    async with _server(peer.handle) as server, http.open():
        model = deepgram.STTv2(
            api_key="test", eot_threshold=0.7, base_url=str(server.make_url("/listen"))
        )
        async with model.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_audio(50))
            assert await peer.received.get() == AUDIO

            model.update_options(eot_threshold=0.85, keyterm=["LiveKit"], language_hint=["en"])
            configure = await peer.received.get()
        await model.aclose()

    assert configure == {
        "type": "Configure",
        "thresholds": {"eot_threshold": 0.85},
        "keyterms": ["LiveKit"],
        "language_hints": ["en"],
    }
    assert peer.sockets.qsize() == 1


async def test_flush_finalizes_the_turn_when_no_audio_is_left_to_send() -> None:
    """A flush that leaves no partial chunk still sends Finalize.

    https://github.com/livekit/agents/pull/6598
    """
    peer = _Peer()
    async with _server(peer.handle) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        async with model.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_audio(50))  # exactly one 50 ms chunk
            assert await peer.received.get() == AUDIO

            stream.flush()

            assert await peer.received.get() == {"type": "Finalize"}
        await model.aclose()


async def test_a_dropped_connection_reconnects_and_later_audio_reaches_the_new_socket() -> None:
    """A socket lost mid-audio is replaced instead of leaving the stream silent.

    https://github.com/livekit/agents/pull/6467 https://github.com/livekit/agents/pull/7206
    https://github.com/livekit/agents/issues/6473
    """
    connections = 0
    second_audio = asyncio.Event()

    async def peer(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if message.type != web.WSMsgType.BINARY:
                continue
            if connections == 1:
                assert request.transport is not None
                request.transport.close()  # gone without a close frame
            else:
                second_audio.set()
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        retry = APIConnectOptions(max_retry=1, retry_interval=0)
        async with model.stream(conn_options=retry) as stream:
            stream.push_frame(_audio(50))
            await until(lambda: connections == 2, "the stream reconnected")
            stream.push_frame(_audio(50))
            await until(second_audio.is_set, "later audio reached the new socket")
        await model.aclose()

    assert connections == 2


async def test_vad_events_pair_start_and_end_of_speech() -> None:
    """SpeechStarted and UtteranceEnd become START_OF_SPEECH and END_OF_SPEECH."""

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.5})
        await ws.send_json({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 1.5})
        async for _ in ws:
            pass
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.STT(
            api_key="test", base_url=str(server.make_url("/listen")), utterance_end_ms=1000
        )
        async with model.stream(conn_options=NO_RETRY) as stream:
            events = [await anext(stream), await anext(stream)]
        await model.aclose()

    assert [event.type for event in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_live_transcript_preserves_word_text_times_and_confidence() -> None:
    """Words keep their punctuated text, stream-relative times and confidence.

    https://github.com/livekit/agents/issues/7180 https://github.com/livekit/agents/pull/6699
    https://github.com/livekit/agents/issues/779
    """

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(_result())
        async for _ in ws:
            pass
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.STT(
            api_key="test", language="multi", base_url=str(server.make_url("/listen"))
        )
        async with model.stream(conn_options=NO_RETRY) as stream:
            final = await anext(
                e async for e in stream if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            )
            offset = stream.start_time_offset
        await model.aclose()

    data = final.alternatives[0]
    assert data.text == "Hello, world!"
    assert (data.start_time, data.end_time) == pytest.approx((1.25 + offset, 2.75 + offset))
    assert data.words is not None
    assert [(w.text, w.confidence) for w in data.words] == [("Hello,", 0.7), ("world!", 0.9)]
    assert [w.start_time for w in data.words] == pytest.approx([1.25 + offset, 2.0 + offset])


async def test_update_options_reconnects_and_the_next_turn_is_transcribed() -> None:
    """Changing the language replaces the socket and recognition keeps working.

    https://github.com/livekit/agents/issues/2109 https://github.com/livekit/agents/issues/2271
    """
    peer = _Peer(reply_to_finalize=True)
    async with _server(peer.handle) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        async with model.stream(conn_options=NO_RETRY) as stream:
            first = await peer.sockets.get()

            model.update_options(language="fr")
            await peer.sockets.get()
            stream.push_frame(_audio(50))
            stream.flush()
            final = await anext(
                e async for e in stream if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            )
        await model.aclose()

    assert first.closed
    assert final.alternatives[0].text == "Hello, world!"
    assert (await peer.queries.get())["language"] == "en-US"
    assert (await peer.queries.get())["language"] == "fr"


@pytest.mark.parametrize("status", [400, 429])
async def test_a_rejected_handshake_raises_with_the_provider_status(status: int) -> None:
    """https://github.com/livekit/agents/issues/1350 https://github.com/livekit/agents/pull/3579"""

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=status, text="rejected")

    async with _server(reject) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        with pytest.raises(APIStatusError) as caught:
            async with model.stream(conn_options=NO_RETRY) as stream:
                await anext(stream)
        await model.aclose()

    assert caught.value.status_code == status


@pytest.mark.parametrize("opt_out", [False, True])
async def test_batch_request_carries_keyterms_and_the_opt_out(opt_out: bool) -> None:
    """https://github.com/livekit/agents/pull/4144 https://github.com/livekit/agents/issues/6059"""
    received = []

    async def recognize(request: web.Request) -> web.Response:
        received.append((request.query.getall("keyterm"), request.query["mip_opt_out"]))
        await request.read()
        return web.json_response({"results": {"channels": [_result()["channel"]]}})

    terms = ["Live Kit", "C++ & Rust"]
    async with _server(recognize, method="POST") as server, http.open():
        model = deepgram.STT(
            api_key="test",
            model="nova-3",
            keyterm=terms,
            mip_opt_out=opt_out,
            base_url=str(server.make_url("/listen")),
        )
        result = await model.recognize(_audio(10))
        await model.aclose()

    assert result.alternatives[0].text == "Hello, world!"
    assert received == [(terms, str(opt_out).lower())]


async def test_an_idle_stream_sends_keepalives_not_audio_and_closes_its_socket() -> None:
    """https://github.com/livekit/agents/issues/736 https://github.com/livekit/agents/pull/3648"""
    received: asyncio.Queue[str] = asyncio.Queue()
    closed = asyncio.Event()

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            received.put_nowait(message.json()["type"])  # a binary frame fails here
        closed.set()
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        async with model.stream():
            first = await received.get()
        await closed.wait()
        await model.aclose()

    assert first == "KeepAlive"
    assert {received.get_nowait() for _ in range(received.qsize())} <= {"KeepAlive", "CloseStream"}


async def test_nova3_keyterms_use_the_singular_query_key_for_any_language() -> None:
    """https://github.com/livekit/agents/issues/1533 https://github.com/livekit/agents/issues/4633"""
    peer = _Peer()
    terms = ["LiveKit", "São Paulo"]
    async with _server(peer.handle) as server, http.open():
        for language in ("multi", "pt"):
            model = deepgram.STT(
                api_key="test",
                model="nova-3",
                language=language,
                keyterm=terms,
                base_url=str(server.make_url("/listen")),
            )
            async with model.stream(conn_options=NO_RETRY):
                query = await peer.queries.get()
            await model.aclose()

            assert query.getall("keyterm") == terms
            assert "keyterms" not in query and "keywords" not in query
            assert query["language"] == language


def test_keywords_and_keyterms_are_refused_on_the_wrong_model_family() -> None:
    """https://github.com/livekit/agents/pull/1898 https://github.com/livekit/agents/pull/3786"""
    with pytest.raises(ValueError, match="Keyterm"):
        deepgram.STT(api_key="test", model="nova-3", keywords=[("LiveKit", 2.0)])
    with pytest.raises(ValueError, match="Nova-3"):
        deepgram.STT(api_key="test", model="nova-2-general", keyterm=["LiveKit"])


async def test_usage_covers_the_connection_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported audio covers the whole billed socket lifetime, not just pushed frames.

    https://github.com/livekit/agents/issues/3464 https://github.com/livekit/agents/issues/5498
    """
    # the plugin times the socket with the wall clock: the test moves it instead of waiting
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(
        "livekit.plugins.deepgram.stt.time", SimpleNamespace(perf_counter=lambda: clock.now)
    )
    audio_arrived = asyncio.Event()

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            audio_arrived.set()
            if message.type == web.WSMsgType.TEXT and message.json()["type"] == "CloseStream":
                await ws.close()
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.STT(api_key="test", base_url=str(server.make_url("/listen")))
        async with model.stream(conn_options=NO_RETRY) as stream:
            stream.push_frame(_audio(100))
            await audio_arrived.wait()  # the socket is open
            clock.now += 0.5  # the socket stays open, and billed, while idle
            stream.end_input()
            usage = [
                event.recognition_usage.audio_duration
                async for event in stream
                if event.recognition_usage is not None
                and event.recognition_usage.audio_duration is not None
            ]
        await model.aclose()

    assert sum(usage) >= 0.5


async def test_a_silent_handshake_fails_within_the_read_timeout() -> None:
    """https://github.com/livekit/agents/issues/1944"""
    accepted: list[asyncio.StreamWriter] = []

    async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.append(writer)

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with http.open():
        model = deepgram.STT(api_key="test", base_url=f"ws://127.0.0.1:{port}/listen")
        options = APIConnectOptions(max_retry=0, read_timeout=0.2)
        with pytest.raises(APITimeoutError):
            async with model.stream(conn_options=options) as stream:
                await anext(stream)
        await model.aclose()
    for writer in accepted:
        writer.close()
    server.close()
    await server.wait_closed()


async def test_language_detection_runs_on_batch_and_is_refused_for_streaming() -> None:
    """https://github.com/livekit/agents/issues/3142"""
    received = []

    async def recognize(request: web.Request) -> web.Response:
        received.append(request.query)
        await request.read()
        channel = _result(detected_language="fr")["channel"]
        return web.json_response({"results": {"channels": [channel]}})

    async with _server(recognize, method="POST") as server, http.open():
        model = deepgram.STT(
            api_key="test", detect_language=True, base_url=str(server.make_url("/listen"))
        )
        with pytest.raises(ValueError, match="language detection"):
            model.stream()
        event = await model.recognize(_audio(10))
        await model.aclose()

    assert event.alternatives[0].language == "fr"
    assert received[0]["detect_language"] == "true"
    assert "language" not in received[0]


def test_flux_eager_threshold_above_the_end_of_turn_threshold_fails_locally() -> None:
    """Deepgram rejects this at handshake, so it fails at construction and update.

    https://github.com/livekit/agents/issues/5199
    """
    with pytest.raises(ValueError, match="eot_threshold"):
        deepgram.STTv2(api_key="test", eager_eot_threshold=0.85)
    model = deepgram.STTv2(api_key="test", eager_eot_threshold=0.5)
    with pytest.raises(ValueError, match="eot_threshold"):
        model.update_options(eager_eot_threshold=0.85)
