"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import json
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import tts as agents_tts, utils
from livekit.plugins.elevenlabs import tts as elevenlabs_tts

pytestmark = pytest.mark.plugin("elevenlabs")


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
        self._input_queue = utils.aio.Chan[object]()
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

    def unregister_stream(self, context_id: str) -> None:
        elevenlabs_tts._Connection.unregister_stream(self, context_id)  # pyright: ignore[reportArgumentType]

    def _cleanup_context(self, context_id: str) -> None:
        elevenlabs_tts._Connection._cleanup_context(self, context_id)  # pyright: ignore[reportArgumentType]

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


@pytest.mark.asyncio
async def test_recv_loop_drops_audio_for_unregistered_context() -> None:
    """Audio flushed after the stream ended must not reach its emitter."""
    context_id = "ctx_123"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(b"late-audio").decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )
    connection.unregister_stream(context_id)

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == []
    # the server released the context, so the connection can drain
    assert connection._active_contexts == set()


def test_unregister_stream_keeps_the_context_closable() -> None:
    """close_context() must still reach the server, otherwise contexts leak (#5844)."""
    context_id = "ctx_123"
    connection = _FakeConnection(context_id, [])
    connection.unregister_stream(context_id)

    assert context_id not in connection._context_data
    assert context_id in connection._active_contexts

    elevenlabs_tts._Connection.close_context(connection, context_id)  # pyright: ignore[reportArgumentType]
    assert connection._input_queue.recv_nowait() == elevenlabs_tts._CloseContext(context_id)


@pytest.mark.asyncio
async def test_interrupted_stream_unregisters_before_ending_the_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #6929: a cancelled run must stop routing audio to its emitter."""
    calls: list[str] = []

    class _StubConnection:
        def register_stream(self, stream: object, emitter: object, waiter: object) -> None:
            pass

        def send_content(self, content: object) -> None:
            calls.append("send_content")

        def unregister_stream(self, context_id: str) -> None:
            calls.append("unregister_stream")

        def close_context(self, context_id: str) -> None:
            calls.append("close_context")

    connection = _StubConnection()

    async def _current_connection(self: object) -> tuple[object, float, bool]:
        return connection, 0.0, True

    original_end_segment = agents_tts.AudioEmitter.end_segment

    def _end_segment(self: agents_tts.AudioEmitter) -> None:
        calls.append("end_segment")
        original_end_segment(self)

    monkeypatch.setattr(elevenlabs_tts.TTS, "_current_connection", _current_connection)
    monkeypatch.setattr(agents_tts.AudioEmitter, "end_segment", _end_segment)

    tts = elevenlabs_tts.TTS(api_key="test-key")
    stream = tts.stream()
    stream.push_text("hello world. ")
    await asyncio.sleep(0.1)  # let the run reach `await waiter`
    await stream.aclose()  # the interruption

    assert calls.index("unregister_stream") < calls.index("end_segment")
    assert "close_context" in calls
