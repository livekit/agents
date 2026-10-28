"""Whisper and Qwen3-ASR deployments have different audio and stop messages."""

import asyncio
import base64
from dataclasses import replace

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
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
