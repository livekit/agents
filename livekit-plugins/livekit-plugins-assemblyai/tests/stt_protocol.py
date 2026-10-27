"""Universal Streaming v3: Begin, audio, Turn, Terminate, Termination, close."""

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Universal(Protocol):
    async def open(self, request):
        assert request.headers["Authorization"]
        assert request.query["sample_rate"] == "16000"
        self.configured = True
        self.connection.send({"type": "Begin", "id": "test-session", "expires_at": 9999999999})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        if event["type"] == "Terminate":
            peer.end_input()
            await peer.finalize()
            peer.send(
                {"type": "Termination", "audio_duration_seconds": 1, "session_duration_seconds": 1},
                final=True,
            )
            peer.close()
        elif event["type"] == "ForceEndpoint":
            await peer.finalize()
        else:
            assert event["type"] in ("UpdateConfiguration", "KeepAlive")

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "Turn",
                "turn_order": result.index,
                "transcript": result.text,
                "end_of_turn": result.final,
                "turn_is_formatted": result.final,
                "words": [
                    {
                        "text": result.text,
                        "start": result.start * 1000,
                        "end": result.end * 1000,
                        "confidence": 1,
                        "word_is_final": result.final,
                    }
                ],
                "end_of_turn_confidence": 1 if result.final else 0,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.assemblyai import STT

    return STT(api_key="test-key", base_url=server.ws_url, sample_rate=16000)


PROVIDERS = (
    Provider(
        "assemblyai",
        Universal,
        create_stt,
        ("https://www.assemblyai.com/docs/streaming/message-sequence",),
        transport_fixture=TransportFixture(
            "assemblyai",
            plugin("assemblyai.stt"),
            completion_messages=(
                {
                    "type": "Turn",
                    "transcript": "words",
                    "end_of_turn": True,
                    "turn_is_formatted": True,
                    "turn_order": 0,
                    "words": [],
                },
            ),
            stop_message={"type": "Terminate"},
        ),
        live_constructor="livekit.plugins.assemblyai.stt:STT",
        required_env=("ASSEMBLYAI_API_KEY",),
        live_options={},
        live_options_from_env={},
        completion_on_close=False,
    ),
)
