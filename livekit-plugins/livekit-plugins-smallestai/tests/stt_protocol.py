"""Pulse: finalize keeps input open; close_stream produces is_last, then close."""

import aiohttp

from tests.stt_harness.faults import FaultCase, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Pulse(Protocol):
    async def open(self, request):
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            event = json_message(message)
            if event["type"] == "close_stream":
                peer.end_input()
                await peer.finalize()
                peer.send(
                    {
                        "type": "transcription",
                        "transcript": "",
                        "is_final": True,
                        "is_last": True,
                        "session_id": "test-session",
                    },
                    final=True,
                )
                peer.close()
            else:
                assert event["type"] == "finalize"
                await peer.finalize()

    async def transcript(self, result):
        self.connection.send(
            {
                "type": "transcription",
                "status": "success",
                "session_id": "test-session",
                "transcript": result.text,
                "is_final": result.final,
                "is_last": False,
                "language": "en",
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.smallestai import STT

    return STT(api_key="test-key", base_url=server.url, sample_rate=16000)


PROVIDERS = (
    Provider(
        "smallestai",
        Pulse,
        create_stt,
        (
            "https://docs.smallest.ai/models/documentation/speech-to-text-pulse/realtime-web-socket/response-format",
        ),
        fault_case=FaultCase(
            "smallestai",
            plugin("smallestai.stt"),
            completion=({"transcript": "words", "is_final": True, "is_last": True},),
            stop={"type": "close_stream"},
        ),
        live_constructor="livekit.plugins.smallestai.stt:STT",
        required_env=("SMALLEST_API_KEY",),
        live_options={},
        live_env={},
        completion_on_close=False,
    ),
)
