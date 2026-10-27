"""Provider wire details for the common WebSocket STT contract tests.

Each case uses the public STT constructor and stream(). Only HTTP/WebSocket I/O is
fake. Add a case here when a new stream implementation uses WebSocketMonitor.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal


@dataclass(frozen=True)
class STTCase:
    name: str
    module: str
    stream_class: str = "SpeechStream"
    constructor: str = "STT"
    options: dict[str, Any] = field(default_factory=dict)
    greetings: tuple[dict[str, Any], ...] = ()
    retry_policy: Literal["standard", "no_replay", "single_attempt"] = "standard"
    completion: tuple[dict[str, Any] | bytes, ...] = ()
    progress: dict[str, Any] | bytes = field(kw_only=True)
    constructor_module: str | None = None
    reply: Callable | None = None
    end_packet: str | None = None
    completion_closes_socket: bool = True
    api_key: bool = True
    http_session: bool = True
    required_env: tuple[str, ...] = ()
    live_options: dict[str, Any] | None = None
    live_env: dict[str, str] = field(default_factory=dict)

    def create(self, session):
        module = import_module(self.constructor_module or self.module)
        kwargs = dict(self.options)
        if self.api_key:
            kwargs["api_key"] = "test-key"
        if self.http_session:
            kwargs["http_session"] = session
        return getattr(module, self.constructor)(**kwargs)

    def create_live(self):
        import os

        module = import_module(self.constructor_module or self.module)
        kwargs = dict(self.options if self.live_options is None else self.live_options)
        kwargs.update({argument: os.environ[name] for argument, name in self.live_env.items()})
        return getattr(module, self.constructor)(**kwargs)

    @property
    def stream_type(self):
        return getattr(import_module(self.module), self.stream_class)


def plugin(name):
    return f"livekit.plugins.{name}"


def _gladia_reply(socket, data):
    if json.loads(data).get("type") == "stop_recording":
        for event in socket.transport.case.completion:
            socket.emit(event)


def _microsoft_reply(socket, data):
    if json.loads(data).get("type") == "input_audio_buffer.commit":
        item_id = f"item-{len(socket.writes)}"
        socket.emit({"type": "input_audio_buffer.committed", "item_id": item_id})
        socket.emit(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": "words",
            }
        )


def _rtzr_reply(socket, data):
    if data == "EOS":
        socket.server_close()


CASES = [
    STTCase(
        "inference",
        "livekit.agents.inference.stt",
        options={
            "model": "deepgram/nova-3",
            "api_secret": "test-secret-" * 4,
            "base_url": "http://unused",
        },
        progress={"type": "start_of_speech"},
        completion=({"type": "final_transcript", "transcript": "words", "language": "en"},),
        required_env=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
        live_options={"model": "deepgram/nova-3"},
    ),
    STTCase(
        "assemblyai",
        plugin("assemblyai.stt"),
        progress={"type": "SpeechStarted"},
        completion=(
            {
                "type": "Turn",
                "transcript": "words",
                "end_of_turn": True,
                "turn_is_formatted": True,
                "turn_order": 0,
                "words": [],
            },
        ),
        required_env=("ASSEMBLYAI_API_KEY",),
    ),
    STTCase(
        "baseten",
        plugin("baseten.stt"),
        completion=({"type": "transcription", "transcript": "words", "is_final": True},),
        options={"model_endpoint": "wss://unused"},
        progress={"type": "transcription", "transcript": "words", "is_final": False},
        required_env=("BASETEN_API_KEY", "BASETEN_MODEL_ENDPOINT"),
        live_options={"model": "whisper"},
    ),
    STTCase(
        "baseten_qwen",
        plugin("baseten.qwen3_stt"),
        "Qwen3SpeechStream",
        options={"model_endpoint": "wss://unused", "model": "qwen3-asr"},
        constructor_module=plugin("baseten.stt"),
        progress={"type": "transcription", "segments": [{"text": "words"}], "is_final": False},
        completion=(
            {
                "type": "transcription",
                "segments": [{"text": "words"}],
                "is_final": True,
                "is_end_of_audio_flush": True,
            },
        ),
        required_env=("BASETEN_API_KEY", "BASETEN_QWEN_MODEL_ENDPOINT"),
        live_options={"model": "qwen3-asr"},
        live_env={"model_endpoint": "BASETEN_QWEN_MODEL_ENDPOINT"},
    ),
    STTCase(
        "cartesia_legacy",
        plugin("cartesia._recognize_streams.legacy_recognize_stream"),
        "LegacyRecognizeStream",
        options={"model": "ink-whisper", "language": "en"},
        constructor_module=plugin("cartesia.stt"),
        progress={"type": "transcript", "text": "words", "is_final": False, "duration": 0.1},
        completion=({"type": "transcript", "text": "words", "is_final": True, "duration": 0.1},),
        required_env=("CARTESIA_API_KEY",),
    ),
    STTCase(
        "cartesia",
        plugin("cartesia._recognize_streams.auto_finalize_recognize_stream"),
        "AutoFinalizeRecognizeStream",
        options={"model": "ink-2", "language": "en"},
        constructor_module=plugin("cartesia.stt"),
        progress={"type": "turn.start"},
        completion=({"type": "turn.start"}, {"type": "turn.end", "transcript": "words"}),
        required_env=("CARTESIA_API_KEY",),
    ),
    STTCase(
        "deepgram",
        plugin("deepgram.stt"),
        progress={"type": "SpeechStarted"},
        completion=(
            {
                "type": "Results",
                "metadata": {"request_id": "test"},
                "is_final": True,
                "speech_final": True,
                "start": 0,
                "duration": 0.1,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": "words",
                            "confidence": 1,
                            "words": [{"word": "words", "start": 0, "end": 0.1, "confidence": 1.0}],
                        }
                    ]
                },
            },
        ),
        required_env=("DEEPGRAM_API_KEY",),
    ),
    STTCase(
        "deepgram_v2",
        plugin("deepgram.stt_v2"),
        "SpeechStreamv2",
        "STTv2",
        progress={"type": "TurnInfo", "event": "StartOfTurn", "transcript": "words", "words": []},
        completion=(
            {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "", "words": []},
            {
                "type": "TurnInfo",
                "event": "EndOfTurn",
                "transcript": "words",
                "words": [{"word": "words", "start": 0, "end": 0.1, "confidence": 1.0}],
            },
        ),
        required_env=("DEEPGRAM_API_KEY",),
    ),
    STTCase(
        "elevenlabs",
        plugin("elevenlabs.stt"),
        options={"use_realtime": True},
        progress={"message_type": "partial_transcript", "text": "words"},
        completion=(
            {
                "message_type": "committed_transcript_with_timestamps",
                "text": "words",
                "words": [{"text": "words", "start": 0, "end": 0.1}],
            },
        ),
        required_env=("ELEVEN_API_KEY",),
    ),
    STTCase(
        "fireworksai",
        plugin("fireworksai.stt"),
        progress={"segments": [{"id": 0, "text": "words", "start": 0, "end": 0.1}]},
        completion=(
            {
                "segments": [
                    {
                        "id": 0,
                        "text": "words",
                        "start": 0,
                        "end": 0.1,
                        "words": [
                            {
                                "word": "words",
                                "start": 0,
                                "end": 0.1,
                                "confidence": 1.0,
                                "is_final": True,
                            }
                        ],
                    }
                ]
            },
        ),
        required_env=("FIREWORKS_API_KEY",),
    ),
    STTCase(
        "gladia",
        plugin("gladia.stt"),
        reply=_gladia_reply,
        end_packet="stop_recording",
        completion_closes_socket=False,
        completion=(
            {
                "type": "transcript",
                "data": {
                    "is_final": True,
                    "utterance": {"text": "words"},
                },
            },
            {"type": "end_session"},
        ),
        progress={
            "type": "transcript",
            "data": {"is_final": False, "utterance": {"text": "words"}},
        },
        required_env=("GLADIA_API_KEY",),
    ),
    STTCase(
        "gradium",
        plugin("gradium.stt"),
        end_packet="end_of_stream",
        completion_closes_socket=False,
        completion=(
            {"type": "text", "text": "words", "start_s": 0},
            {"type": "end_text", "stop_s": 0.1},
            {"type": "end_of_stream"},
        ),
        greetings=({"type": "ready"},),
        progress={"type": "text", "text": "words", "start_s": 0},
        required_env=("GRADIUM_API_KEY",),
    ),
    STTCase(
        "inworld",
        plugin("inworld.stt"),
        progress={"result": {"speechStarted": {}}},
        completion=({"result": {"transcription": {"transcript": "words", "isFinal": True}}},),
        required_env=("INWORLD_API_KEY",),
    ),
    STTCase(
        "meta",
        plugin("meta.stt"),
        greetings=({"sessionId": "test"},),
        retry_policy="no_replay",
        progress={"type": "audioProgress", "audioProcessedMs": 100},
        completion=(
            {"type": "speechStart", "turnId": "test"},
            {"type": "speechEnd", "turnId": "test"},
            {"type": "speechComplete", "turnId": "test", "transcript": "words"},
        ),
        required_env=("META_API_KEY",),
    ),
    STTCase(
        "microsoft_ai",
        plugin("microsoft_ai.stt"),
        reply=_microsoft_reply,
        options={
            "vad": None,
            "url": "wss://unused",
            "model": "test",
        },
        greetings=({"type": "session.created"}, {"type": "session.updated"}),
        completion=(
            {"type": "input_audio_buffer.committed", "item_id": "test"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "test",
                "transcript": "words",
            },
        ),
        retry_policy="no_replay",
        progress={
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "test",
            "delta": "words",
        },
        required_env=("MICROSOFT_AI_STT_API_KEY", "MICROSOFT_AI_STT_URL", "MICROSOFT_AI_STT_MODEL"),
        live_options={"vad": None},
    ),
    STTCase(
        "nabrah",
        plugin("nabrah.stt"),
        greetings=({"type": "ready"},),
        retry_policy="no_replay",
        progress={"type": "transcript", "text": "words", "is_final": False, "audio_processed": 0.1},
        completion=(
            {"type": "transcript", "text": "words <eot>", "is_final": True, "audio_processed": 0.2},
        ),
        required_env=("NABRAH_API_KEY",),
    ),
    STTCase(
        "openai",
        plugin("openai.stt"),
        options={"use_realtime": True},
        http_session=False,
        progress={"type": "input_audio_buffer.speech_started"},
        completion=(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "test",
                "transcript": "words",
            },
        ),
        required_env=("OPENAI_API_KEY",),
    ),
    STTCase(
        "rtzr",
        plugin("rtzr.stt"),
        reply=_rtzr_reply,
        api_key=False,
        progress={
            "alternatives": [{"text": "words"}],
            "final": False,
            "start_at": 0,
            "duration": 100,
        },
        completion=(
            {"alternatives": [{"text": "words"}], "final": True, "start_at": 0, "duration": 100},
        ),
        required_env=("RTZR_CLIENT_ID", "RTZR_CLIENT_SECRET"),
        live_options={"language": "en"},
    ),
    STTCase(
        "sarvam",
        plugin("sarvam.stt"),
        retry_policy="single_attempt",
        progress={
            "type": "data",
            "data": {"transcript": "words", "language_code": "en-IN", "request_id": "test"},
        },
        completion=(
            {
                "type": "data",
                "data": {"transcript": "words", "language_code": "en-IN", "request_id": "test"},
            },
        ),
        required_env=("SARVAM_API_KEY",),
        live_options={"language": "en-IN"},
    ),
    STTCase(
        "sarvam_realtime",
        plugin("sarvam.stt_streaming"),
        "RealtimeSpeechStream",
        "STTRealtime",
        retry_policy="single_attempt",
        progress={"event": "vad.speech_start"},
        completion=(
            {"event": "vad.speech_start", "utterance_idx": 0},
            {"event": "transcript.final", "utterance_idx": 0, "text": "words", "language": "en-IN"},
            {"event": "vad.speech_end", "utterance_idx": 0},
        ),
        required_env=("SARVAM_API_KEY",),
        live_options={"language": "en-IN"},
    ),
    STTCase(
        "simplismart",
        plugin("simplismart.stt"),
        options={"streaming": True},
        progress=b"words",
        completion=(b"words",),
        required_env=("SIMPLISMART_API_KEY",),
    ),
    STTCase(
        "slng",
        plugin("slng.stt"),
        options={
            "model": "deepgram/nova-3",
        },
        greetings=({"type": "ready"},),
        progress={"type": "partial_transcript", "transcript": "words"},
        completion=({"type": "final_transcript", "transcript": "words"},),
        required_env=("SLNG_API_KEY",),
    ),
    STTCase(
        "smallestai",
        plugin("smallestai.stt"),
        progress={"type": "speech_started"},
        completion=({"transcript": "words", "is_final": True, "is_last": True},),
        required_env=("SMALLEST_API_KEY",),
    ),
    STTCase(
        "soniox",
        plugin("soniox.stt"),
        progress={"tokens": [{"text": "words", "is_final": False}]},
        completion=(
            {
                "tokens": [
                    {
                        "word": "words",
                        "start": 0,
                        "end": 0.1,
                        "confidence": 1.0,
                        "text": "words",
                        "is_final": True,
                        "language": "en",
                    },
                    {"text": "<end>", "is_final": True},
                ],
                "total_audio_proc_ms": 200,
            },
            {"tokens": [], "finished": True},
        ),
        required_env=("SONIOX_API_KEY",),
    ),
    STTCase(
        "telnyx",
        plugin("telnyx.stt"),
        progress={"transcript": "words", "is_final": False},
        completion=({"transcript": "words", "is_final": True},),
        required_env=("TELNYX_API_KEY",),
    ),
    STTCase(
        "xai",
        plugin("xai.stt"),
        greetings=({"type": "transcript.created"},),
        progress={"type": "transcript.partial", "text": "words"},
        completion=({"type": "transcript.done", "text": "words"},),
        required_env=("XAI_API_KEY",),
    ),
]


CASES_BY_NAME = {case.name: case for case in CASES}


def make_provider(case, monkeypatch):
    import aiohttp

    from livekit.agents import utils

    from .fake_stt_transport import Transport

    transport = Transport(case)

    async def unexpected_request(*args, **kwargs):
        raise AssertionError("STT contract tests must not use the network")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unexpected_request)
    monkeypatch.setenv("RTZR_CLIENT_ID", "test")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test")
    monkeypatch.setattr(aiohttp, "ClientSession", type(transport.session()))
    monkeypatch.setattr(utils.http_context, "http_session", transport.session)
    instance = case.create(transport.session())
    return case, instance, transport


async def assert_control_write_failure(case, monkeypatch, message, action):
    """Check provider-specific stop/flush packets with the shared transport."""
    import asyncio
    import json

    import aiohttp
    import pytest

    from livekit import rtc
    from livekit.agents import APIConnectionError, APIConnectOptions

    from .fake_stt_transport import Socket

    cause = aiohttp.ClientConnectionError("connection reset")
    original_send = Socket.send_str

    async def send(self, data):
        event = json.loads(data)
        if message in event or event.get("type") == message:
            raise cause
        await original_send(self, data)

    monkeypatch.setattr(Socket, "send_str", send)
    _, provider, _ = make_provider(case, monkeypatch)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        if action == "audio":
            stream.push_frame(rtc.AudioFrame.create(16000, 1, 160))
        elif action == "flush":
            stream.flush()
        else:
            stream.end_input()
        with pytest.raises(APIConnectionError) as info:
            await asyncio.wait_for(stream._task, 1)
        assert info.value.retryable
        assert info.value.__cause__ is cause
    finally:
        await stream.aclose()
        await provider.aclose()
