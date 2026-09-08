"""Keyless Maya protocol tests using an in-memory WebSocket, with no network."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIError, APIStatusError, tokenize, tts
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import maya

pytestmark = pytest.mark.unit
OPTIONS = APIConnectOptions(max_retry=0, timeout=0.2)
PCM = b"\x21\x03\x32\x04" * 2400
METADATA = {"type": "metadata", "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}


class Socket:
    def __init__(self, service: Service) -> None:
        self.service = service
        self.closed = False
        self.queue: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
        self.frames: list[dict[str, Any]] = []
        self.active: dict[str, bool] = {}

    def reply(self, data: Any) -> None:
        self.queue.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data)))

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)
        self.service.sent.set()
        context = frame.get("context_id")
        if frame["type"] == "start":
            if self.service.start_send_error:
                raise ConnectionError("unit-secret-do-not-log")
            if self.service.bad_json:
                self.queue.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{broken"))
            else:
                self.reply(self.service.metadata)
            return
        if frame["type"] == "cancel":
            self.active.pop(context, None)
            self.reply({"type": "cancelled", "context_id": context})
            return
        if frame["type"] != "text":
            raise AssertionError("Unexpected request type")
        text = frame["text"]
        self.active[context] = not frame["continue"]
        if text.strip():
            if self.service.mode == "error":
                self.reply(
                    {"type": "error", "error": "unit-secret-do-not-log", "context_id": context}
                )
                return
            if self.service.mode == "global-error":
                self.reply({"type": "error", "error": "unit-secret-do-not-log"})
                return
            if self.service.mode == "cancelled":
                self.reply({"type": "cancelled", "context_id": context})
                return
            if self.service.mode == "no-response":
                return
            if self.service.mode == "bad-audio":
                self.reply({"type": "audio", "context_id": context, "audio": "%%%not-base64"})
                return
            if self.service.mode != "empty":
                for extra in self.service.extra_frames:
                    self.reply(extra)
                import base64

                for chunk in self.service.chunks:
                    self.reply(
                        {
                            "type": "audio",
                            "context_id": context,
                            "audio": base64.b64encode(chunk).decode(),
                        }
                    )
            if self.service.mode == "disconnect":
                self.queue.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))
                return
            if self.service.mode == "early-end":
                self.reply({"type": "end", "context_id": context})
                return
        if not frame["continue"] and self.service.mode not in ("hold", "no-response", "disconnect"):
            self.reply({"type": "end", "context_id": context})
            self.active.pop(context, None)

    async def receive(self, *, timeout: float | None) -> SimpleNamespace:
        return await asyncio.wait_for(self.queue.get(), timeout)

    async def close(self) -> None:
        self.closed = True


class Service:
    """An HTTP-session-shaped fake; every websocket and request stays in memory."""

    def __init__(self, **kwargs: Any) -> None:
        self.metadata: Any = dict(METADATA)
        self.mode = "normal"
        self.bad_json = False
        self.start_send_error = False
        self.connect_status: int | None = None
        self.chunks = [PCM]
        self.extra_frames: list[dict[str, Any]] = []
        self.sockets: list[Socket] = []
        self.headers: list[dict[str, str]] = []
        self.urls: list[str] = []
        self.closed = False
        self.sent = asyncio.Event()
        for name, value in kwargs.items():
            setattr(self, name, value)

    async def ws_connect(self, url: str, *, headers: dict[str, str]) -> Any:
        self.headers.append(headers)
        self.urls.append(url)
        if self.connect_status:
            raise aiohttp.ClientResponseError(
                request_info=cast(Any, None),
                history=(),
                status=self.connect_status,
                message="unit-secret-do-not-log",
            )
        socket = Socket(self)
        self.sockets.append(socket)
        return socket

    def engine(self, **kwargs: Any) -> maya.TTS:
        return maya.TTS(
            api_key="unit-key",
            http_session=cast(aiohttp.ClientSession, self),
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=1, stream_context_len=1),
            **kwargs,
        )

    def text_frames(self) -> list[dict[str, Any]]:
        return [f for socket in self.sockets for f in socket.frames if f["type"] == "text"]

    def cancels(self) -> list[dict[str, Any]]:
        return [f for socket in self.sockets for f in socket.frames if f["type"] == "cancel"]


async def collect(stream: tts.ChunkedStream | tts.SynthesizeStream) -> bytes:
    pcm = bytearray()
    try:
        async for event in stream:
            assert event.frame.sample_rate == 24000
            assert event.frame.num_channels == 1
            pcm.extend(event.frame.data)
    finally:
        await stream.aclose()
    return bytes(pcm)


async def speak(engine: maya.TTS, streaming: bool, text: str = "Hello there.") -> bytes:
    if not streaming:
        return await collect(engine.synthesize(text, conn_options=OPTIONS))
    stream = engine.stream(conn_options=OPTIONS)
    stream.push_text(text)
    stream.end_input()
    return await collect(stream)


@pytest.fixture
async def service() -> AsyncIterator[Service]:
    yield Service()


def test_brand_and_explicit_current_defaults() -> None:
    engine = Service().engine()
    assert engine.provider == "Maya Research"
    assert engine.model == "Maya Calyx"
    assert engine._settings.start() == {
        "type": "start",
        "v2": True,
        "model": "Maya Calyx",
        "voice": "Aarav",
    }


def test_future_model_and_voice_are_not_hardcoded() -> None:
    engine = Service().engine(
        model="future-model", voice="future-voice", language="future-language"
    )
    assert engine._settings.start()["language"] == "future-language"
    assert engine.model == "future-model"


@pytest.mark.parametrize("name", ["model", "voice", "language"])
@pytest.mark.parametrize("value", ["", " ", "bad\nvalue"])
def test_invalid_configuration_is_rejected(name: str, value: str) -> None:
    with pytest.raises(ValueError):
        Service().engine(**{name: value})


def test_key_argument_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_API_KEY", "environment-key")
    assert maya.TTS()._api_key == "environment-key"
    assert maya.TTS(api_key="explicit")._api_key == "explicit"
    monkeypatch.delenv("MAYA_API_KEY")
    with pytest.raises(ValueError, match="MAYA_API_KEY"):
        maya.TTS()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "https://user:password@example.com",
        "https://example.com?key=secret",
        "https://example.com#secret",
        "not-a-url",
    ],
)
def test_base_url_rejects_unsafe_or_unsupported_shapes(url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        Service().engine(base_url=url)


@pytest.mark.parametrize("streaming", [False, True])
async def test_valid_audio_and_handshake(service: Service, streaming: bool) -> None:
    async with service.engine(language="hi") as engine:
        audio = await speak(engine, streaming)
    assert audio == PCM
    assert service.headers == [{"Authorization": "Bearer unit-key"}]
    assert service.urls == ["wss://tts.mayaresearch.ai/v1/tts/stream"]
    assert service.sockets[0].frames[0]["language"] == "hi"
    frames = service.text_frames()
    assert len({f["context_id"] for f in frames}) == 1
    assert sum(not f["continue"] for f in frames) == 1
    assert not service.cancels()
    assert service.sockets[0].closed and not service.closed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("text", ["", " ", "\n\t"])
async def test_blank_turn_never_connects(service: Service, streaming: bool, text: str) -> None:
    async with service.engine() as engine:
        assert await speak(engine, streaming, text) == b""
    assert not service.sockets


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_rate", 48000),
        ("sample_rate", 0),
        ("sample_rate", None),
        ("sample_rate", "24000"),
        ("channels", 2),
        ("channels", True),
        ("channels", None),
        ("encoding", "mulaw"),
        ("encoding", None),
    ],
)
async def test_bad_audio_metadata_fails_before_text(
    service: Service,
    streaming: bool,
    field: str,
    value: Any,
) -> None:
    service.metadata[field] = value
    async with service.engine() as engine:
        with pytest.raises(APIError, match="audio format"):
            await speak(engine, streaming)
    assert not service.text_frames()
    assert all(socket.closed for socket in service.sockets)


@pytest.mark.parametrize(
    "metadata", [[], None, {"type": "error", "error": "unit-secret-do-not-log"}]
)
async def test_rejected_or_nonobject_handshake_is_safe(metadata: Any) -> None:
    service = Service(metadata=metadata)
    async with service.engine() as engine:
        with pytest.raises(APIError) as result:
            await speak(engine, False)
    assert "unit-secret" not in str(result.value)
    assert not service.text_frames() and service.sockets[0].closed


async def test_malformed_handshake_closes_socket() -> None:
    service = Service(bad_json=True)
    async with service.engine() as engine:
        with pytest.raises(APIError, match="malformed JSON"):
            await speak(engine, False)
    assert service.sockets[0].closed


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_http_status_does_not_leak_headers(status: int) -> None:
    service = Service(connect_status=status)
    async with service.engine() as engine:
        with pytest.raises(APIStatusError) as result:
            await speak(engine, False)
    assert result.value.status_code == status
    assert "unit-secret" not in str(result.value)
    if status in (401, 403):
        assert result.value.retryable is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "chunks",
    [
        [PCM[:1], PCM[1:17], PCM[17:]],
        [PCM[:317], PCM[317:]],
        [PCM[:2]],
        [PCM[:634]],
        [PCM[:2518]],
    ],
)
async def test_split_samples_and_partial_tail_are_preserved(
    streaming: bool, chunks: list[bytes]
) -> None:
    service = Service(chunks=chunks)
    async with service.engine() as engine:
        received = await speak(engine, streaming)
    expected = b"".join(chunks)
    assert received[: len(expected)] == expected
    assert not any(received[len(expected) :])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["bad-audio", "error", "global-error", "empty"])
async def test_protocol_errors_are_not_retried_or_leaked(streaming: bool, mode: str) -> None:
    service = Service(mode=mode)
    async with service.engine() as engine:
        with pytest.raises(APIError) as result:
            await speak(engine, streaming)
    assert not result.value.retryable
    assert "unit-secret" not in str(result.value)
    assert len(service.sockets) == 1 and service.sockets[0].closed


@pytest.mark.parametrize("streaming", [False, True])
async def test_truncated_final_sample_is_an_error(streaming: bool) -> None:
    service = Service(chunks=[PCM + b"x"])
    async with service.engine() as engine:
        with pytest.raises(APIError, match="truncated") as result:
            await speak(engine, streaming)
    assert not result.value.retryable


@pytest.mark.parametrize("streaming", [False, True])
async def test_late_or_unscoped_frames_cannot_be_spoken(service: Service, streaming: bool) -> None:
    service.extra_frames = [
        {"type": "audio", "context_id": "old", "audio": "CQAJAA=="},
        {"type": "end", "context_id": "old"},
        {"type": "cancelled", "context_id": "old"},
        {"type": "error", "context_id": "old"},
        {"type": "audio", "audio": "CQAJAA=="},
        {"type": "end"},
        {"type": "pong"},
    ]
    async with service.engine() as engine:
        assert await speak(engine, streaming) == PCM


async def test_completed_turns_reuse_one_socket_and_distinct_contexts(service: Service) -> None:
    async with service.engine() as engine:
        for _ in range(3):
            assert await speak(engine, False) == PCM
    assert len(service.sockets) == 1
    assert len({f["context_id"] for f in service.text_frames()}) == 3


async def test_slow_first_text_is_not_a_server_timeout(service: Service) -> None:
    async with service.engine() as engine:
        stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
        receiver = asyncio.create_task(collect(stream))
        await asyncio.sleep(0.1)
        assert not receiver.done() and not service.sockets
        stream.push_text("Hello after waiting.")
        stream.end_input()
        assert await asyncio.wait_for(receiver, 1) == PCM


async def test_sentences_share_a_context_and_keep_word_boundaries(service: Service) -> None:
    async with service.engine() as engine:
        stream = engine.stream(conn_options=OPTIONS)
        stream.push_text("This is the first sentence. ")
        stream.push_text("This is the next sentence.")
        stream.end_input()
        assert await collect(stream)
    texts = [f["text"] for f in service.text_frames() if f["text"].strip()]
    assert all(text.endswith(" ") for text in texts)
    assert (
        " ".join("".join(texts).split()) == "This is the first sentence. This is the next sentence."
    )
    assert len({f["context_id"] for f in service.text_frames()}) == 1


async def test_startup_transport_failure_does_not_expose_secrets(service: Service) -> None:
    service.start_send_error = True
    async with service.engine() as engine:
        with pytest.raises(APIError) as exc:
            await speak(engine, False)
        assert "unit-secret" not in str(exc.value)
        assert service.sockets[0].closed


@pytest.mark.parametrize("streaming", [False, True])
async def test_missing_audio_times_out_and_cancels(service: Service, streaming: bool) -> None:
    service.mode = "no-response"
    async with service.engine() as engine:
        with pytest.raises(APIError):
            await asyncio.wait_for(speak(engine, streaming), 1)
        assert service.sockets[0].closed
        assert len(service.cancels()) == 1


async def test_stale_frames_cannot_keep_a_dead_turn_alive(service: Service) -> None:
    service.mode = "no-response"
    async with service.engine() as engine:
        task = asyncio.create_task(speak(engine, False))
        await service.sent.wait()

        async def stale_frames() -> None:
            while not task.done():
                service.sockets[0].reply({"type": "pong"})
                service.sockets[0].reply({"type": "end", "context_id": "old-turn"})
                await asyncio.sleep(0.01)

        stale = asyncio.create_task(stale_frames())
        try:
            with pytest.raises(APIError):
                await asyncio.wait_for(task, 1)
        finally:
            await stale
        assert service.sockets[0].closed


async def test_concurrent_turns_never_share_an_active_socket(service: Service) -> None:
    service.mode = "hold"
    async with service.engine() as engine:
        first = engine.synthesize("First active turn.", conn_options=OPTIONS)
        second = engine.synthesize("Second active turn.", conn_options=OPTIONS)
        await asyncio.wait_for(asyncio.gather(anext(first), anext(second)), 1)
        assert len(service.sockets) == 2
        assert len({f["context_id"] for f in service.text_frames()}) == 2
        await first.aclose()
        assert not service.sockets[1].closed
        await second.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_cancel_discards_socket_and_next_turn_is_fresh(
    service: Service, streaming: bool
) -> None:
    service.mode = "hold"
    async with service.engine() as engine:
        if streaming:
            stream = engine.stream(conn_options=OPTIONS)
            stream.push_text("Please interrupt this sentence. ")
            stream.flush()
        else:
            stream = engine.synthesize("Please interrupt.", conn_options=OPTIONS)
        await asyncio.wait_for(anext(stream), 1)
        await stream.aclose()
        assert service.cancels()
        assert service.sockets[0].closed
        service.mode = "normal"
        assert await speak(engine, False) == PCM
    assert len(service.sockets) == 2
    assert service.cancels()[0]["context_id"] != service.text_frames()[-1]["context_id"]


async def test_cancelled_terminal_does_not_wait_for_llm_to_finish(service: Service) -> None:
    service.mode = "cancelled"
    async with service.engine() as engine:
        stream = engine.stream(conn_options=OPTIONS)
        stream.push_text("Server will cancel this. ")
        stream.flush()
        # No end_input: a cancelled terminal must close the receiver AND sender.
        assert await asyncio.wait_for(collect(stream), 1) == b""
    assert not service.cancels()


async def test_early_end_while_input_is_open_is_not_success(service: Service) -> None:
    service.mode = "early-end"
    async with service.engine() as engine:
        stream = engine.stream(conn_options=OPTIONS)
        stream.push_text("Still waiting for more input. ")
        stream.flush()
        with pytest.raises(APIError, match="before text input"):
            await asyncio.wait_for(collect(stream), 1)


async def test_update_options_keeps_active_turn_and_replaces_next_socket(service: Service) -> None:
    service.mode = "hold"
    async with service.engine(language="hi") as engine:
        old = engine.synthesize("Old voice.", conn_options=OPTIONS)
        await asyncio.wait_for(anext(old), 1)
        engine.update_options(model="future-model", voice="future-voice", language="te")
        assert not service.sockets[0].closed
        service.mode = "normal"
        assert await speak(engine, False) == PCM
        assert service.sockets[1].frames[0]["voice"] == "future-voice"
        assert service.sockets[1].frames[0]["model"] == "future-model"
        assert not service.sockets[0].closed
        await old.aclose()


async def test_prewarmed_old_settings_are_not_used_for_new_turn(service: Service) -> None:
    async with service.engine() as engine:
        engine.prewarm()
        await asyncio.wait_for(service.sent.wait(), 1)
        engine.update_options(voice="Tarini")
        assert await speak(engine, False) == PCM
        assert service.sockets[-1].frames[0]["voice"] == "Tarini"
        assert not any(f["type"] == "text" for f in service.sockets[0].frames)


async def test_disconnect_after_pcm_never_replays_the_turn() -> None:
    service = Service(mode="disconnect")
    async with service.engine() as engine:
        with pytest.raises(APIError) as result:
            await collect(
                engine.synthesize(
                    "Do not repeat me.", conn_options=APIConnectOptions(max_retry=2, timeout=0.2)
                )
            )
    assert not result.value.retryable
    assert len(service.text_frames()) == 1


async def test_close_stops_active_stream_and_preserves_caller_session(service: Service) -> None:
    service.mode = "hold"
    engine = service.engine()
    stream = engine.synthesize("Close this active stream.", conn_options=OPTIONS)
    await asyncio.wait_for(anext(stream), 1)
    await engine.aclose()
    await engine.aclose()
    assert all(socket.closed for socket in service.sockets)
    assert not service.closed
    with pytest.raises(APIError, match="closed"):
        await speak(engine, False)


@pytest.mark.parametrize("url", ["http://example.com", "ws://example.com", "http://localhost"])
def test_plaintext_base_url_is_rejected_before_connecting(url: str) -> None:
    service = Service()
    with pytest.raises(ValueError, match="base_url"):
        service.engine(base_url=url)
    assert not service.headers and not service.sockets


async def test_acquisition_finishing_after_close_cannot_return_a_socket(
    service: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    original = service.ws_connect

    async def blocked_connect(url: str, *, headers: dict[str, str]) -> Any:
        entered.set()
        await release.wait()
        return await original(url, headers=headers)

    monkeypatch.setattr(service, "ws_connect", blocked_connect)
    engine = service.engine()

    async def borrow() -> None:
        async with engine._connection(0.5):
            pass

    task = asyncio.create_task(borrow())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await engine.aclose()
        release.set()
        with pytest.raises(APIError, match="closed"):
            await asyncio.wait_for(task, 1)
        assert all(socket.closed for socket in service.sockets)
        assert not service.text_frames()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await engine.aclose()


async def test_pause_between_sentences_is_not_a_maya_timeout(service: Service) -> None:
    async with service.engine() as engine:
        stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
        # A partial next sentence gives the tokenizer lookahead without flush(),
        # which would close this LiveKit segment to further push_text calls.
        stream.push_text("First sentence. Second ")
        await asyncio.wait_for(anext(stream), 1)
        await asyncio.sleep(0.12)
        assert not stream._task.done(), "A pause awaiting the LLM aborted the Maya turn"
        stream.push_text("sentence.")
        stream.end_input()
        assert await asyncio.wait_for(collect(stream), 1)
    texts = [f["text"] for f in service.text_frames() if f["text"].strip()]
    assert " ".join("".join(texts).split()) == "First sentence. Second sentence."
    assert len({f["context_id"] for f in service.text_frames()}) == 1


@pytest.mark.parametrize("resume", ["text", "closer"])
async def test_after_input_pause_provider_wait_is_bounded_again(
    service: Service, resume: str
) -> None:
    async with service.engine() as engine:
        stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
        stream.push_text("First sentence. Second ")
        await asyncio.wait_for(anext(stream), 1)
        await asyncio.sleep(0.12)
        assert not stream._task.done()
        service.mode = "no-response"
        if resume == "text":
            stream.push_text("sentence. Third ")
        else:
            stream.end_input()
        with pytest.raises(APIError, match="timed out"):
            await asyncio.wait_for(collect(stream), 1)
        assert service.cancels() and service.sockets[0].closed


async def test_repeated_input_cannot_extend_an_unanswered_response(service: Service) -> None:
    service.mode = "no-response"
    async with service.engine() as engine:
        stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.06))

        async def feed() -> None:
            while True:
                stream.push_text("Another complete sentence. ")
                await asyncio.sleep(0.01)

        sender = asyncio.create_task(feed())
        try:
            with pytest.raises(APIError, match="timed out"):
                await asyncio.wait_for(collect(stream), 0.8)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        assert len(service.text_frames()) > 1
        assert service.sockets[0].closed


async def test_cancellation_while_waiting_for_more_input_closes_receive(service: Service) -> None:
    async with service.engine() as engine:
        stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
        stream.push_text("First sentence. Second ")
        await asyncio.wait_for(anext(stream), 1)
        await asyncio.sleep(0.12)
        assert not stream._task.done()
        await asyncio.wait_for(stream.aclose(), 1)
        assert service.cancels() and service.sockets[0].closed
        assert await speak(engine, False) == PCM


@pytest.mark.parametrize("streaming", [False, True])
async def test_blocked_text_send_is_bounded(
    service: Service, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    original = Socket.send_json

    async def blocked_send(socket: Socket, frame: dict[str, Any]) -> None:
        if frame["type"] == "text":
            await asyncio.Event().wait()
        await original(socket, frame)

    monkeypatch.setattr(Socket, "send_json", blocked_send)
    async with service.engine() as engine:
        with pytest.raises(APIError, match="timed out"):
            await asyncio.wait_for(speak(engine, streaming), 1)
        assert service.cancels() and service.sockets[0].closed


@pytest.mark.parametrize(
    "mode", ["one-shot", "stream", "direct-one-shot", "direct-stream", "prewarm"]
)
async def test_public_shutdown_cancels_inflight_connect(
    service: Service, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    original = service.ws_connect

    async def blocked_connect(url: str, *, headers: dict[str, str]) -> Any:
        entered.set()
        await release.wait()
        return await original(url, headers=headers)

    monkeypatch.setattr(service, "ws_connect", blocked_connect)
    engine = service.engine()
    task = None
    if mode == "prewarm":
        engine.prewarm()
    else:
        if mode == "one-shot":
            stream = engine.synthesize("Never send this.", conn_options=OPTIONS)
        elif mode == "direct-one-shot":
            stream = maya.ChunkedStream(
                tts=engine, input_text="Never send this.", conn_options=OPTIONS
            )
        else:
            stream = (
                maya.SynthesizeStream(tts=engine, conn_options=OPTIONS)
                if mode == "direct-stream"
                else engine.stream(conn_options=OPTIONS)
            )
            stream.push_text("Never send this.")
            stream.end_input()
        task = asyncio.create_task(collect(stream))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(engine.aclose(), 1)
        release.set()
        if task is not None:
            await asyncio.wait_for(task, 1)
        assert not service.text_frames()
        assert all(socket.closed for socket in service.sockets)
        assert not engine._pool._available
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await engine.aclose()


@pytest.mark.parametrize("url", ["https://example.com/prefix/", "wss://example.com/prefix/"])
async def test_encrypted_base_url_retains_path_prefix(service: Service, url: str) -> None:
    async with service.engine(base_url=url) as engine:
        assert await speak(engine, False) == PCM
    assert service.urls == ["wss://example.com/prefix/v1/tts/stream"]


def test_plaintext_environment_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAYA_BASE_URL", "http://example.com")
    with pytest.raises(ValueError, match="HTTPS/WSS"):
        Service().engine()


async def test_clear_language_omits_next_start_without_interrupting_active_turn(
    service: Service,
) -> None:
    service.mode = "hold"
    async with service.engine(language="hi") as engine:
        old = engine.synthesize("An active Hindi turn.", conn_options=OPTIONS)
        await asyncio.wait_for(anext(old), 1)
        engine.update_options(language=None)
        assert not service.sockets[0].closed
        assert service.sockets[0].frames[0]["language"] == "hi"
        service.mode = "normal"
        assert await speak(engine, False) == PCM
        assert "language" not in service.sockets[1].frames[0]
        assert not service.sockets[0].closed
        await old.aclose()


def test_none_constructor_language_omits_the_field() -> None:
    assert "language" not in Service().engine(language=None)._settings.start()


def test_not_given_update_leaves_language_unchanged() -> None:
    engine = Service().engine(language="hi")
    engine.update_options(voice="Tarini", language=NOT_GIVEN)
    assert engine._settings.start()["language"] == "hi"
    engine.update_options(model="future-model")
    assert engine._settings.start()["language"] == "hi"


@pytest.mark.parametrize("name", ["model", "voice", "language"])
@pytest.mark.parametrize("value", ["", " ", "bad\nvalue"])
def test_invalid_option_update_is_atomic(name: str, value: str) -> None:
    engine = Service().engine(language="hi")
    original = engine._settings
    with pytest.raises(ValueError):
        engine.update_options(**{name: value})
    assert engine._settings == original


async def test_clearing_an_already_omitted_language_reuses_connection(service: Service) -> None:
    async with service.engine() as engine:
        assert await speak(engine, False) == PCM
        engine.update_options(language=None)
        assert await speak(engine, False) == PCM
        assert len(service.sockets) == 1
