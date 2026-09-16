"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import json
import logging
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


# -- eleven_v3 / eleven_v3_conversational (text-to-dialogue) --------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("eleven_v3", True),
        ("eleven_v3_conversational", True),
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
