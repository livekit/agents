"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.agents import tts as agents_tts, utils
from livekit.plugins.elevenlabs import tts as elevenlabs_tts

pytestmark = pytest.mark.plugin("elevenlabs")


class _FakeWebSocket:
    def __init__(self, messages: list[object], *, close_code: int = 1000) -> None:
        self._messages = messages
        self.closed = False
        self.close_code = close_code

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


@pytest.mark.asyncio
async def test_prewarm_opens_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = asyncio.Event()

    async def _current_connection(
        self: object,
    ) -> tuple[SimpleNamespace, float, bool]:
        opened.set()
        return SimpleNamespace(_recv_task=None), 0.0, False

    monkeypatch.setattr(elevenlabs_tts.TTS, "_current_connection", _current_connection)

    tts = elevenlabs_tts.TTS(api_key="test-key")
    tts.prewarm()
    await asyncio.wait_for(opened.wait(), timeout=1)
    await tts.aclose()


@pytest.mark.asyncio
async def test_option_update_during_connect_discards_stale_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_started = asyncio.Event()
    resume_connect = asyncio.Event()
    connections: list[object] = []

    class _StubConnection:
        def __init__(self, opts: object, session: object) -> None:
            self._opts = opts
            self._closed = False
            self.is_current = True
            self._recv_task = None
            connections.append(self)

        async def connect(self) -> None:
            if len(connections) == 1:
                connect_started.set()
                await resume_connect.wait()

        async def aclose(self) -> None:
            self._closed = True

        def mark_non_current(self) -> None:
            self.is_current = False

    monkeypatch.setattr(elevenlabs_tts, "_Connection", _StubConnection)

    async with aiohttp.ClientSession() as session:
        tts = elevenlabs_tts.TTS(api_key="test-key", voice_id="old", http_session=session)
        connection_task = asyncio.create_task(tts._current_connection())
        await asyncio.wait_for(connect_started.wait(), timeout=1)

        tts.update_options(voice_id="new")
        resume_connect.set()
        connection, _, _ = await asyncio.wait_for(connection_task, timeout=1)

        assert len(connections) == 2
        assert connections[0]._closed  # pyright: ignore[reportAttributeAccessIssue]
        assert connection is connections[1]
        assert connection._opts.voice_id == "new"

        await tts.aclose()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_prewarm_stops_on_non_retryable_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    finished = asyncio.Event()

    async def _current_connection(self: object) -> tuple[SimpleNamespace, float, bool]:
        nonlocal attempts
        attempts += 1
        raise elevenlabs_tts.APIStatusError("Unauthorized", status_code=401)

    monkeypatch.setattr(elevenlabs_tts.TTS, "_current_connection", _current_connection)
    monkeypatch.setattr(
        elevenlabs_tts.asyncio,
        "sleep",
        lambda _delay: finished.set(),
    )

    tts = elevenlabs_tts.TTS(api_key="test-key")
    tts.prewarm()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert attempts == 1
    assert not finished.is_set()
    await tts.aclose()


async def test_prewarm_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    ready = asyncio.Event()

    async def _current_connection(
        self: object,
    ) -> tuple[SimpleNamespace, float, bool]:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise ConnectionError("temporary failure")
        ready.set()
        return SimpleNamespace(_recv_task=None), 0.0, False

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(elevenlabs_tts.TTS, "_current_connection", _current_connection)
    monkeypatch.setattr(elevenlabs_tts.asyncio, "sleep", _sleep)

    tts = elevenlabs_tts.TTS(api_key="test-key")
    tts.prewarm()
    await asyncio.wait_for(ready.wait(), timeout=1)
    await tts.aclose()

    assert attempts == 4
    assert delays == [1.0, 2.0, 4.0]


async def test_dialogue_prewarm_reconnects_when_idle_socket_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[asyncio.Future[None]] = []
    reconnected = asyncio.Event()

    async def _current_connection(
        self: object,
    ) -> tuple[SimpleNamespace, float, bool]:
        recv_task = asyncio.get_running_loop().create_future()
        connections.append(recv_task)
        if len(connections) == 2:
            reconnected.set()
        return SimpleNamespace(_recv_task=recv_task), 0.0, False

    monkeypatch.setattr(elevenlabs_tts.TTS, "_current_connection", _current_connection)

    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v4")
    tts.prewarm()
    while not connections:
        await asyncio.sleep(0)

    connections[0].cancel()
    await asyncio.wait_for(reconnected.wait(), timeout=1)
    await tts.aclose()

    assert len(connections) == 2
    assert not connections[1].cancelled()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection_cls", [elevenlabs_tts._Connection, elevenlabs_tts._DialogueConnection]
)
@pytest.mark.parametrize("input_closed", [False, True])
async def test_recv_loop_resets_timeout_timer_on_audio(
    connection_cls: type, input_closed: bool
) -> None:
    context_id = "ctx_123"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(b"hello-audio").decode("ascii"),
                }
            ),
        ],
    )
    ctx = connection._context_data[context_id]
    ctx.input_closed = input_closed
    timer = asyncio.get_event_loop().call_later(60, lambda: None)
    ctx.timeout_timer = timer
    restarted: list[str] = []
    connection._start_timeout_timer = restarted.append  # type: ignore[attr-defined]

    with contextlib.suppress(Exception):
        await connection_cls._recv_loop(connection)

    # cleared so _start_timeout_timer can arm a new timer on the next send
    assert timer.cancelled()
    assert ctx.timeout_timer is None
    # after close_context, no send will re-arm it: the audio handler must
    assert restarted == ([context_id] if input_closed else [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_cls", "model"),
    [
        (elevenlabs_tts._Connection, "eleven_flash_v2_5"),
        (elevenlabs_tts._DialogueConnection, "eleven_v3_conversational"),
    ],
)
async def test_send_loop_arms_timeout_on_close_context(connection_cls: type, model: str) -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model=model, voice_id="voice-1")
    async with aiohttp.ClientSession() as session:
        connection = connection_cls(tts._opts, session)
        connection._ws = _RecordingWs()
        ctx = elevenlabs_tts._StreamData(
            emitter=_FakeEmitter(),  # type: ignore[arg-type]
            stream=SimpleNamespace(_conn_options=SimpleNamespace(timeout=60)),  # type: ignore[arg-type]
            waiter=asyncio.get_event_loop().create_future(),
        )
        connection._context_data["ctx-1"] = ctx
        connection._active_contexts.add("ctx-1")
        connection.close_context("ctx-1")
        connection._input_queue.close()

        await asyncio.wait_for(connection._send_loop(), timeout=1.0)

        # no more sends will arm it, so close_context must
        assert ctx.input_closed
        assert ctx.timeout_timer is not None
        ctx.timeout_timer.cancel()


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


# -- eleven_v3 / eleven_v4 (text-to-dialogue) ----------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("eleven_v3", True),
        ("eleven_v3_conversational", True),
        ("eleven_v4", True),
        ("eleven_v4_turbo", True),
        ("eleven_turbo_v2_5", False),
        ("eleven_flash_v2_5", False),
    ],
)
def test_is_dialogue_model(model: str, expected: bool) -> None:
    assert elevenlabs_tts.is_dialogue_model(model) is expected


def test_dialogue_synthesize_url_targets_text_to_dialogue_endpoint() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational")
    url = elevenlabs_tts._dialogue_synthesize_url(tts._opts)  # pyright: ignore[reportPrivateUsage]

    assert url.startswith(f"{elevenlabs_tts.API_BASE_URL_V1}/text-to-dialogue/stream?")
    assert "voice_id" not in url


def test_dialogue_multi_stream_url_omits_regular_tts_only_params() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        voice_id="voice-1",
        enable_ssml_parsing=True,
        chunk_length_schedule=[80, 120],
    )
    url = elevenlabs_tts._dialogue_multi_stream_url(  # pyright: ignore[reportPrivateUsage]
        tts._opts
    )

    assert url.startswith("wss://")
    assert "/text-to-dialogue/multi-stream-input?" in url
    assert "voice-1" not in url
    assert "model_id=eleven_v3_conversational" in url
    assert "enable_ssml_parsing" not in url
    assert "inactivity_timeout" not in url
    assert "auto_mode" not in url


def test_build_dialogue_synthesize_body_single_turn() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        voice_id="voice-1",
        pronunciation_dictionary_locators=[
            elevenlabs_tts.PronunciationDictionaryLocator(
                pronunciation_dictionary_id="dict-1",
                version_id="v1",
            )
        ],
    )
    body = elevenlabs_tts._build_dialogue_synthesize_body(  # pyright: ignore[reportPrivateUsage]
        tts._opts, "hello there", voice_settings=None
    )

    assert body["inputs"] == [{"text": "hello there", "voice_id": "voice-1"}]
    assert body["model_id"] == "eleven_v3_conversational"
    assert "settings" not in body
    assert body["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "dict-1", "version_id": "v1"}
    ]


def test_build_dialogue_synthesize_body_keeps_only_supported_settings() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    body = elevenlabs_tts._build_dialogue_synthesize_body(  # pyright: ignore[reportPrivateUsage]
        tts._opts, "hello there", voice_settings={"stability": 0.5, "similarity_boost": 0.75}
    )

    assert body["settings"] == {"stability": 0.5}


def test_build_dialogue_context_init_packet_registers_single_voice() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    packet = elevenlabs_tts._build_dialogue_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet == {"context_id": "ctx-1", "voices": ["voice-1"]}


def test_build_dialogue_context_init_packet_keeps_only_supported_voice_settings() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        voice_id="voice-1",
        voice_settings=elevenlabs_tts.VoiceSettings(stability=0.5, similarity_boost=0.75),
    )
    packet = elevenlabs_tts._build_dialogue_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet["voice_settings"] == {"stability": 0.5}


def test_dialogue_model_warns_on_unsupported_voice_settings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        elevenlabs_tts.TTS(
            api_key="test-key",
            model="eleven_v3_conversational",
            voice_settings=elevenlabs_tts.VoiceSettings(stability=0.5, similarity_boost=0.75),
        )

    assert any("voice_settings.similarity_boost" in r.getMessage() for r in caplog.records)


def test_build_dialogue_context_init_packet_includes_pronunciation_dictionaries() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        voice_id="voice-1",
        pronunciation_dictionary_locators=[
            elevenlabs_tts.PronunciationDictionaryLocator(
                pronunciation_dictionary_id="dict-1",
                version_id="v1",
            )
        ],
    )
    packet = elevenlabs_tts._build_dialogue_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "dict-1", "version_id": "v1"}
    ]


@pytest.mark.asyncio
async def test_dialogue_recv_loop_parses_audio_and_alignment() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "alignment": {
                        "chars": ["h", "i"],
                        "char_start_times_ms": [0, 100],
                        "char_durations_ms": [100, 100],
                    },
                    "is_final": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._DialogueConnection._recv_loop(  # pyright: ignore[reportPrivateUsage]
        connection
    )

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.emitter.timed_transcript_pushes >= 1
    assert connection.waiter.done()
    assert connection.waiter.result() is None
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_dialogue_recv_loop_turn_boundary_does_not_resolve_waiter() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "is_final_audio_for_turn": True,
                }
            ),
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "is_final": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._DialogueConnection._recv_loop(  # pyright: ignore[reportPrivateUsage]
        connection
    )

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None


@pytest.mark.asyncio
async def test_dialogue_recv_loop_reports_error() -> None:
    context_id = "ctx_123"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "error": "something went wrong",
                }
            ),
        ],
    )

    await elevenlabs_tts._DialogueConnection._recv_loop(  # pyright: ignore[reportPrivateUsage]
        connection
    )

    assert connection.waiter.done()
    exc = connection.waiter.exception()
    assert isinstance(exc, elevenlabs_tts.APIError)
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_dialogue_recv_loop_drops_audio_for_unregistered_context() -> None:
    context_id = "ctx_123"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(b"late-audio").decode("ascii"),
                    "is_final": True,
                }
            ),
        ],
    )
    connection.unregister_stream(context_id)

    await elevenlabs_tts._DialogueConnection._recv_loop(  # pyright: ignore[reportPrivateUsage]
        connection
    )

    assert connection.emitter.audio_chunks == []
    assert connection._active_contexts == set()


class _RecordingWs:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send_json(self, data: dict[str, object]) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_dialogue_send_loop_sends_close_context_without_waiting() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        ws = _RecordingWs()
        connection._ws = ws  # type: ignore[assignment]
        connection.send_content(
            elevenlabs_tts._SynthesizeContent("ctx-1", "hello ", flush=True)  # pyright: ignore[reportPrivateUsage]
        )
        connection.close_context("ctx-1")
        connection._input_queue.close()

        await asyncio.wait_for(connection._send_loop(), timeout=1.0)

        assert ws.sent == [
            {"context_id": "ctx-1", "voices": ["voice-1"]},
            {
                "context_id": "ctx-1",
                "inputs": [{"text": "hello ", "voice_id": "voice-1"}],
                "flush": True,
            },
            {"context_id": "ctx-1", "close_context": True},
        ]


@pytest.mark.asyncio
async def test_dialogue_send_loop_sends_keep_alive_for_idle_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_tts, "_DIALOGUE_KEEP_ALIVE_INTERVAL", 0.05)
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        ws = _RecordingWs()
        connection._ws = ws  # type: ignore[assignment]
        connection.send_content(
            elevenlabs_tts._SynthesizeContent("ctx-1", "hello ", flush=True)  # pyright: ignore[reportPrivateUsage]
        )
        send_task = asyncio.create_task(connection._send_loop())

        await asyncio.sleep(0.2)
        assert {"context_id": "ctx-1", "keep_alive": True} in ws.sent

        connection._input_queue.close()
        await asyncio.wait_for(send_task, timeout=1.0)


@pytest.mark.asyncio
async def test_dialogue_send_loop_keeps_idle_context_alive_during_other_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_tts, "_DIALOGUE_KEEP_ALIVE_INTERVAL", 0.05)
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        ws = _RecordingWs()
        connection._ws = ws  # type: ignore[assignment]
        connection.send_content(
            elevenlabs_tts._SynthesizeContent("ctx-idle", "hello ")  # pyright: ignore[reportPrivateUsage]
        )
        send_task = asyncio.create_task(connection._send_loop())

        for _ in range(20):
            connection.send_content(
                elevenlabs_tts._SynthesizeContent("ctx-busy", "hello ")  # pyright: ignore[reportPrivateUsage]
            )
            await asyncio.sleep(0.01)

        assert {"context_id": "ctx-idle", "keep_alive": True} in ws.sent

        connection._input_queue.close()
        await asyncio.wait_for(send_task, timeout=1.0)


@pytest.mark.asyncio
async def test_dialogue_send_loop_stops_keep_alive_once_context_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_tts, "_DIALOGUE_KEEP_ALIVE_INTERVAL", 0.05)
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        ws = _RecordingWs()
        connection._ws = ws  # type: ignore[assignment]
        connection.send_content(
            elevenlabs_tts._SynthesizeContent("ctx-1", "hello ", flush=True)  # pyright: ignore[reportPrivateUsage]
        )
        connection.close_context("ctx-1")
        send_task = asyncio.create_task(connection._send_loop())

        await asyncio.sleep(0.2)
        assert {"context_id": "ctx-1", "keep_alive": True} not in ws.sent
        assert {"context_id": "ctx-1", "close_context": True} in ws.sent

        connection._input_queue.close()
        await asyncio.wait_for(send_task, timeout=1.0)


@pytest.fixture
async def dialogue_connection() -> AsyncIterator[elevenlabs_tts._DialogueConnection]:
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(tts._opts, session)
        try:
            yield connection
        finally:
            await connection.aclose()


_DIALOGUE_ERROR_LOG = "elevenlabs text-to-dialogue returned error"
_DIALOGUE_IDLE_TIMEOUT_LOG = "elevenlabs text-to-dialogue socket idle timeout"


def _dialogue_idle_timeout() -> dict[str, object]:
    return {
        "error": "input_timeout_exceeded",
        "message": "No message received within 20s.",
        "code": 1008,
        "context_id": None,
    }


def _register_dialogue_stream(
    connection: elevenlabs_tts._DialogueConnection,
) -> asyncio.Future[None]:
    stream = SimpleNamespace(
        _context_id="ctx-1",
        _text_buffer="",
        _start_times_ms=[],
        _durations_ms=[],
        _conn_options=SimpleNamespace(timeout=60),
    )
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    connection.register_stream(stream, _FakeEmitter(), waiter)  # type: ignore[arg-type]
    return waiter


def _log_records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.getMessage() == message]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_context_id", [False, True])
async def test_dialogue_idle_timeout_without_streams_logs_debug(
    dialogue_connection: elevenlabs_tts._DialogueConnection,
    caplog: pytest.LogCaptureFixture,
    include_context_id: bool,
) -> None:
    payload = _dialogue_idle_timeout()
    if not include_context_id:
        payload.pop("context_id")
    dialogue_connection._ws = _FakeWebSocket([_websocket_text_message(payload)], close_code=1008)  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG, logger=elevenlabs_tts.logger.name):
        await dialogue_connection._recv_loop()

    records = _log_records(caplog, _DIALOGUE_IDLE_TIMEOUT_LOG)
    assert [record.levelno for record in records] == [logging.DEBUG]
    assert getattr(records[0], "lk.pii.data") == payload
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert dialogue_connection._closed


@pytest.mark.asyncio
@pytest.mark.parametrize("input_closed", [False, True])
async def test_dialogue_idle_timeout_with_registered_stream_logs_error(
    dialogue_connection: elevenlabs_tts._DialogueConnection,
    caplog: pytest.LogCaptureFixture,
    input_closed: bool,
) -> None:
    # registered before its first text packet, or waiting for its final audio
    waiter = _register_dialogue_stream(dialogue_connection)
    if input_closed:
        dialogue_connection._context_data["ctx-1"].input_closed = True
        dialogue_connection._active_contexts.add("ctx-1")
        dialogue_connection._closing_contexts.add("ctx-1")
    dialogue_connection._ws = _FakeWebSocket(
        [_websocket_text_message(_dialogue_idle_timeout())], close_code=1008
    )  # type: ignore[assignment]

    await dialogue_connection._recv_loop()

    records = _log_records(caplog, _DIALOGUE_ERROR_LOG)
    assert [record.levelno for record in records] == [logging.ERROR]
    exc = waiter.exception()
    assert isinstance(exc, elevenlabs_tts.APIStatusError)
    assert exc.status_code == 1008


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"context_id": "ctx-1"},
        {"context_id": ""},
        {"error": "invalid_api_key"},
        {"code": 1000},
        {"code": None},
    ],
)
async def test_dialogue_other_errors_log_error(
    dialogue_connection: elevenlabs_tts._DialogueConnection,
    caplog: pytest.LogCaptureFixture,
    overrides: dict[str, object],
) -> None:
    payload = _dialogue_idle_timeout() | overrides
    dialogue_connection._ws = _FakeWebSocket([_websocket_text_message(payload)])  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG, logger=elevenlabs_tts.logger.name):
        await dialogue_connection._recv_loop()

    records = _log_records(caplog, _DIALOGUE_ERROR_LOG)
    assert [record.levelno for record in records] == [logging.ERROR]
    assert not _log_records(caplog, _DIALOGUE_IDLE_TIMEOUT_LOG)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_dialogue_idle_timeout_after_stream_ends_logs_debug(
    dialogue_connection: elevenlabs_tts._DialogueConnection,
    caplog: pytest.LogCaptureFixture,
    cancelled: bool,
) -> None:
    waiter = _register_dialogue_stream(dialogue_connection)
    dialogue_connection._active_contexts.add("ctx-1")
    messages: list[object] = []
    if cancelled:
        waiter.cancel()
        dialogue_connection.unregister_stream("ctx-1")
    else:
        messages.append(_websocket_text_message({"context_id": "ctx-1", "is_final": True}))
    messages.append(_websocket_text_message(_dialogue_idle_timeout()))
    dialogue_connection._ws = _FakeWebSocket(messages, close_code=1008)  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG, logger=elevenlabs_tts.logger.name):
        await dialogue_connection._recv_loop()

    records = _log_records(caplog, _DIALOGUE_IDLE_TIMEOUT_LOG)
    assert [record.levelno for record in records] == [logging.DEBUG]
    assert not _log_records(caplog, _DIALOGUE_ERROR_LOG)
    assert waiter.cancelled() if cancelled else waiter.result() is None
