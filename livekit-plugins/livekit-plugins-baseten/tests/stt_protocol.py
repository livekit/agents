"""Whisper and Qwen3-ASR deployments have different audio and stop messages."""

import asyncio
import base64
from dataclasses import replace

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider, RequestError
from tests.stt_harness.server import json_message

SOURCE = (
    "https://docs.baseten.co/reference/inference-api/predict-endpoints/streaming-transcription-api"
)


class Whisper(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            config = json_message(message)
            assert set(config) == {"whisper_params", "streaming_params", "streaming_vad_config"}
            assert config["streaming_params"]["sample_rate"] == 16000
            self.configured = True
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"type": "end_audio"}
            peer.end_input()
            await peer.acknowledge({"type": "end_audio", "body": {"status": "acknowledged"}})
            await peer.finalize()
            peer.send({"type": "end_audio", "body": {"status": "finished"}}, final=True)
            peer.close()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "transcription",
                "transcript": result.text,
                "is_final": result.final,
                "language_code": "en",
                "start": result.start,
                "end": result.end,
            },
            final=result.final,
        )


class LegacyWhisper(Whisper):
    async def receive(self, message):
        if self.configured and message.type == aiohttp.WSMsgType.TEXT:
            assert json_message(message) == {"type": "end_audio"}
            self.connection.end_input()
            return
        await super().receive(message)


class Qwen(Whisper):
    def __init__(self, connection):
        super().__init__(connection)
        self.committing = False
        self.flushed = asyncio.Event()

    async def receive(self, message):
        event = json_message(message)
        if not self.configured:
            assert set(event) == {"whisper_params", "streaming_params", "streaming_vad_config"}
            self.configured = True
        elif event["type"] == "input_audio_buffer.append":
            await self.connection.accept_audio(base64.b64decode(event["audio"], validate=True))
        else:
            assert event["type"] == "input_audio_buffer.commit"
            self.committing = True
            # Every commit has a response, including an empty turn.
            await self.send_transcript(self.connection.result(final=True))
            self.connection.server.final_requested.set()
            self.committing = False
            self.flushed.set()

    async def finish_turn(self):
        await asyncio.wait_for(self.flushed.wait(), 1)
        self.flushed.clear()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "transcription",
                "is_final": result.final,
                "is_end_of_audio_flush": result.final and self.committing,
                "language": "English",
                "segments": [{"text": result.text, "start": result.start, "end": result.end}],
            },
            final=result.final,
        )


def create_whisper_stt(server, monkeypatch):
    from livekit.plugins.baseten import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url)


def create_qwen_stt(server, monkeypatch):
    from livekit.plugins.baseten import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url, model="qwen3-asr")


PROVIDERS = (
    Provider(
        "baseten",
        Whisper,
        create_whisper_stt,
        (SOURCE,),
        completion_on_close=False,
        transport_fixture=TransportFixture(
            "baseten",
            plugin("baseten.stt"),
            completion_messages=(
                {"type": "transcription", "transcript": "words", "is_final": True},
                {"type": "end_audio", "body": {"status": "finished"}},
            ),
            options={"model_endpoint": "wss://unused"},
            stop_message={"type": "end_audio"},
        ),
        live_constructor="livekit.plugins.baseten.stt:STT",
        required_env=("BASETEN_API_KEY", "BASETEN_MODEL_ENDPOINT"),
        live_options={"model": "whisper"},
        live_options_from_env={},
    ),
    Provider(
        "baseten_qwen",
        Qwen,
        create_qwen_stt,
        (SOURCE,),
        transport_fixture=TransportFixture(
            "baseten_qwen",
            plugin("baseten.qwen3_stt"),
            "Qwen3SpeechStream",
            options={"model_endpoint": "wss://unused", "model": "qwen3-asr"},
            constructor_module=plugin("baseten.stt"),
            completion_messages=(
                {
                    "type": "transcription",
                    "segments": [{"text": "words"}],
                    "is_final": True,
                    "is_end_of_audio_flush": True,
                },
            ),
            stop_message={"type": "input_audio_buffer.commit"},
        ),
        live_constructor="livekit.plugins.baseten.stt:STT",
        required_env=("BASETEN_API_KEY", "BASETEN_QWEN_MODEL_ENDPOINT"),
        live_options={"model": "qwen3-asr"},
        live_options_from_env={"model_endpoint": "BASETEN_QWEN_MODEL_ENDPOINT"},
        completion_on_close=False,
    ),
)


PROVIDERS += (
    replace(
        PROVIDERS[0],
        name="baseten_legacy",
        protocol=LegacyWhisper,
        completion_unknown_reason="Older Whisper runtimes do not acknowledge end_audio",
        completion_on_close=True,
        transport_fixture=None,
        live_constructor="",
    ),
)


V2_SOURCE = (
    "https://docs.baseten.co/reference/inference-api/predict-endpoints/"
    "streaming-transcription-api-v2"
)
V2_FORMAT = {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}


class StreamingV2(Protocol):
    """Session events, base64 audio, an answer to every flush, and a drained 1000 close."""

    def __init__(self, connection):
        super().__init__(connection)
        self.flushed = asyncio.Event()

    async def open(self, request):
        # The server speaks first on every connection.
        self.connection.send(
            {
                "type": "session.created",
                "session_id": "sess_test",
                "capabilities": {"max_turn_s": None},
                "defaults": {"input_format": V2_FORMAT, "segmentation_mode": "server_vad"},
            }
        )

    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        if not self.configured:
            assert event["type"] == "session.update"
            assert set(event) == {"type", "session"}
            session = event["session"]
            assert session["input_format"] == V2_FORMAT
            assert session["segmentation_mode"] in ("server_vad", "client_flush")
            self.configured = True
            peer.send(
                {
                    "type": "session.updated",
                    "session_id": "sess_test",
                    "granted": session,
                    "refusals": [],
                }
            )
        elif event["type"] == "input_audio_buffer.append":
            assert set(event) == {"type", "audio"}
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif event["type"] == "input_audio_buffer.flush":
            assert set(event) == {"type"}
            # A flush ends the turn and is answered, with an empty transcript too.
            await self.send_transcript(peer.result(final=True))
            peer.server.final_requested.set()
            self.flushed.set()
        else:
            assert event == {"type": "close"}
            # Pending work drains before the 1000 close.
            peer.end_input()
            peer.close()

    async def finish_turn(self):
        await asyncio.wait_for(self.flushed.wait(), 1)
        self.flushed.clear()

    async def send_transcript(self, result):
        segments = (
            [{"text": result.text, "start": result.start, "end": result.end}] if result.text else []
        )
        self.connection.send(
            {
                "type": "transcription.done" if result.final else "transcription.delta",
                "transcript": {"turn_id": result.index, "segments": segments, "language": "en"},
            },
            final=result.final,
        )


def create_streaming_v2_stt(server, monkeypatch):
    from livekit.plugins.baseten import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url, model="streaming-v2")


def _v2_error(code, close_code, retryable):
    return RequestError(
        code,
        body={"type": "error", "code": code, "message": "rejected"},
        close_code=close_code,
        retryable=retryable,
    )


PROVIDERS += (
    Provider(
        "baseten_streaming_v2",
        StreamingV2,
        create_streaming_v2_stt,
        (V2_SOURCE,),
        request_errors=(
            _v2_error("invalid_event", 1008, retryable=False),
            _v2_error("internal_error", 1011, retryable=True),
            _v2_error("at_capacity", 1013, retryable=True),
            RequestError("forbidden", http_status=403),
        ),
        live_constructor="livekit.plugins.baseten.stt:STT",
        required_env=("BASETEN_API_KEY", "BASETEN_STREAMING_V2_MODEL_ENDPOINT"),
        live_options={"model": "streaming-v2"},
        live_options_from_env={"model_endpoint": "BASETEN_STREAMING_V2_MODEL_ENDPOINT"},
    ),
)
