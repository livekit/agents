"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import json
from types import SimpleNamespace

import aiohttp
import pytest

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
        self._turn_drained: dict[str, asyncio.Event] = {}

    def _cleanup_context(self, context_id: str) -> None:
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()
        self._active_contexts.discard(context_id)
        self._turn_drained.pop(context_id, None)

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
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    body = elevenlabs_tts._build_dialogue_synthesize_body(  # pyright: ignore[reportPrivateUsage]
        tts._opts, "hello there", voice_settings=None
    )

    assert body["inputs"] == [{"text": "hello there", "voice_id": "voice-1"}]
    assert body["model_id"] == "eleven_v3_conversational"
    assert "settings" not in body


def test_build_dialogue_synthesize_body_includes_settings_when_given() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key", model="eleven_v3_conversational", voice_id="voice-1"
    )
    body = elevenlabs_tts._build_dialogue_synthesize_body(  # pyright: ignore[reportPrivateUsage]
        tts._opts, "hello there", voice_settings={"stability": 0.5}
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


def test_build_dialogue_context_init_packet_includes_voice_settings_when_given() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        voice_id="voice-1",
        voice_settings=elevenlabs_tts.VoiceSettings(stability=0.5, similarity_boost=0.75),
    )
    packet = elevenlabs_tts._build_dialogue_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75}


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
                    "normalized_alignment": {
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
async def test_wait_for_turn_drained_returns_once_event_is_set() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational")
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        connection._turn_drained["ctx-1"] = asyncio.Event()
        connection._turn_drained["ctx-1"].set()

        await asyncio.wait_for(connection._wait_for_turn_drained("ctx-1"), timeout=1.0)


@pytest.mark.asyncio
async def test_wait_for_turn_drained_falls_through_without_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs_tts, "_CLOSE_CONTEXT_DRAIN_TIMEOUT", 0.05)
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational")
    async with aiohttp.ClientSession() as session:
        connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
            tts._opts, session
        )
        await asyncio.wait_for(connection._wait_for_turn_drained("ctx-missing"), timeout=1.0)
