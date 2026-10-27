"""Whisper and Qwen3-ASR deployments have different audio and stop messages."""

import base64

import aiohttp

from tests.stt_harness.faults import FaultCase, plugin
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

    async def transcript(self, result):
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


class Qwen(Whisper):
    async def receive(self, message):
        event = json_message(message)
        if not self.configured:
            assert set(event) == {"whisper_params", "streaming_params", "streaming_vad_config"}
            self.configured = True
        elif event["type"] == "input_audio_buffer.append":
            await self.connection.accept_audio(base64.b64decode(event["audio"], validate=True))
        else:
            assert event["type"] == "input_audio_buffer.commit"
            await self.connection.finalize()

    async def transcript(self, result):
        self.connection.send(
            {
                "type": "transcription",
                "is_final": result.final,
                "is_end_of_audio_flush": result.final,
                "language": "English",
                "segments": [{"text": result.text, "start": result.start, "end": result.end}],
            },
            final=result.final,
        )


def whisper(server, monkeypatch):
    from livekit.plugins.baseten import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url)


def qwen(server, monkeypatch):
    from livekit.plugins.baseten import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url, model="qwen3-asr")


PROVIDERS = (
    Provider(
        "baseten",
        Whisper,
        whisper,
        (SOURCE,),
        fault_case=FaultCase(
            "baseten",
            plugin("baseten.stt"),
            completion=(
                {"type": "transcription", "transcript": "words", "is_final": True},
                {"type": "end_audio", "body": {"status": "finished"}},
            ),
            options={"model_endpoint": "wss://unused"},
            stop={"type": "end_audio"},
            completion_closes_socket=False,
        ),
        live_constructor="livekit.plugins.baseten.stt:STT",
        required_env=("BASETEN_API_KEY", "BASETEN_MODEL_ENDPOINT"),
        live_options={"model": "whisper"},
        live_env={},
    ),
    Provider(
        "baseten_qwen",
        Qwen,
        qwen,
        (SOURCE,),
        fault_case=FaultCase(
            "baseten_qwen",
            plugin("baseten.qwen3_stt"),
            "Qwen3SpeechStream",
            options={"model_endpoint": "wss://unused", "model": "qwen3-asr"},
            constructor_module=plugin("baseten.stt"),
            completion=(
                {
                    "type": "transcription",
                    "segments": [{"text": "words"}],
                    "is_final": True,
                    "is_end_of_audio_flush": True,
                },
            ),
            completion_closes_socket=False,
            stop={"type": "input_audio_buffer.commit"},
        ),
        live_constructor="livekit.plugins.baseten.stt:STT",
        required_env=("BASETEN_API_KEY", "BASETEN_QWEN_MODEL_ENDPOINT"),
        live_options={"model": "qwen3-asr"},
        live_env={"model_endpoint": "BASETEN_QWEN_MODEL_ENDPOINT"},
        completion_on_close=False,
    ),
)
