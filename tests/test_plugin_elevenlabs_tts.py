"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import json
import socket
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from livekit.plugins.elevenlabs import tts as elevenlabs_tts

pytestmark = pytest.mark.plugin("elevenlabs")


class _FakeJobContext:
    def __init__(self) -> None:
        self.shutdown_callbacks: list[Callable[[], Awaitable[None]]] = []

    def add_shutdown_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self.shutdown_callbacks.append(callback)


class _FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.closed = False

    async def receive(self) -> object:
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data="")

    async def close(self) -> None:
        self.closed = True


class _FakeEmitter:
    def __init__(self) -> None:
        self.audio_chunks: list[bytes] = []
        self.timed_transcript_pushes = 0

    def push(self, audio: bytes) -> None:
        self.audio_chunks.append(audio)

    def push_timed_transcript(self, _timed_words: object) -> None:
        self.timed_transcript_pushes += 1


class _FakeStream:
    def __init__(self) -> None:
        self._text_buffer = ""
        self._start_times_ms: list[int] = []
        self._durations_ms: list[int] = []


class _FakeConnection:
    def __init__(self, context_id: str, messages: list[object]) -> None:
        self._closed = False
        self._ws = _FakeWebSocket(messages)
        self._is_current = True
        self._active_contexts = {context_id}
        self.emitter = _FakeEmitter()
        self.waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._context_data = {
            context_id: elevenlabs_tts._StreamData(
                emitter=self.emitter,
                stream=_FakeStream(),
                waiter=self.waiter,
            )
        }
        self.preferred_alignment = "normalized"

    def _cleanup_context(self, context_id: str) -> None:
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()
        self._active_contexts.discard(context_id)

    async def aclose(self) -> None:
        self._closed = True
        await self._ws.close()


def _websocket_text_message(payload: dict[str, object]) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def test_auto_mode_defaults_to_true_without_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key")
    assert tts._opts.auto_mode is True


def test_auto_mode_defaults_to_false_with_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", chunk_length_schedule=[120, 160, 250, 290])
    assert tts._opts.auto_mode is False


def test_auto_mode_respects_explicit_value_with_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        chunk_length_schedule=[120, 160, 250, 290],
        auto_mode=True,
    )
    assert tts._opts.auto_mode is True


@pytest.mark.asyncio
async def test_job_shutdown_gracefully_closes_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_code: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()

    async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        try:
            async for _ in websocket:
                pass
        finally:
            if not close_code.done():
                close_code.set_result(websocket.close_code)
        return websocket

    app = web.Application()
    app.router.add_get("/{path:.*}", websocket_handler)
    runner = web.AppRunner(app)
    await runner.setup()

    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", 0))
    port = server_socket.getsockname()[1]
    site = web.SockSite(runner, server_socket)
    await site.start()

    job_ctx = _FakeJobContext()
    monkeypatch.setattr(elevenlabs_tts, "get_job_context", lambda *, required=False: job_ctx)

    try:
        async with aiohttp.ClientSession() as http_session:
            tts = elevenlabs_tts.TTS(
                api_key="test-key",
                voice_id="test-voice",
                base_url=f"http://127.0.0.1:{port}",
                http_session=http_session,
            )
            try:
                await tts._current_connection()  # pyright: ignore[reportPrivateUsage]
                await tts._current_connection()  # pyright: ignore[reportPrivateUsage]

                assert len(job_ctx.shutdown_callbacks) == 1
                await job_ctx.shutdown_callbacks[0]()

                assert await asyncio.wait_for(close_code, timeout=1) == 1000
                assert not http_session.closed
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()


def test_build_context_init_packet_includes_generation_config() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", chunk_length_schedule=[80, 120], auto_mode=False)
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet["text"] == " "
    assert packet["context_id"] == "ctx-1"
    assert packet["generation_config"] == {"chunk_length_schedule": [80, 120]}


def test_build_context_init_packet_omits_generation_config_when_not_set() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key")
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-2"
    )

    assert "generation_config" not in packet


def test_build_context_init_packet_includes_pronunciation_dictionaries() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        pronunciation_dictionary_locators=[
            elevenlabs_tts.PronunciationDictionaryLocator(
                pronunciation_dictionary_id="dict-1",
                version_id="v1",
            )
        ],
    )
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-3"
    )

    assert packet["pronunciation_dictionary_locators"] == [
        {
            "pronunciation_dictionary_id": "dict-1",
            "version_id": "v1",
        }
    ]


@pytest.mark.asyncio
async def test_recv_loop_accepts_snake_case_context_id() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_recv_loop_still_accepts_camel_case_context_id() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "contextId": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_recv_loop_ignores_flush_done_for_active_context() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "type": "flush_done",
                    "context_id": context_id,
                    "status_code": 206,
                    "done": False,
                    "data": "",
                    "flush_done": True,
                }
            ),
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None


@pytest.mark.asyncio
async def test_recv_loop_ignores_flush_done_for_inactive_context() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "type": "flush_done",
                    "context_id": "already_closed_context",
                    "status_code": 206,
                    "done": False,
                    "data": "",
                    "flush_done": True,
                }
            ),
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None
