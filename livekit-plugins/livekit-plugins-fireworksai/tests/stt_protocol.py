"""Legacy streaming endpoint: binary PCM and segment/word events.

The former public streaming reference is unavailable. The checkpoint EOF exchange
needs a server trace before it can serve as an independent completion oracle.
"""

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Whisper(Protocol):
    async def open(self, request):
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"checkpoint_id": "final"}
            peer.end_input()
            peer.server.final_requested.set()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "segments": [
                    {
                        "id": result.index,
                        "text": result.text,
                        "start": result.start,
                        "end": result.end,
                        "words": [
                            {
                                "word": result.text,
                                "start": result.start,
                                "end": result.end,
                                "confidence": 1,
                                "is_final": result.final,
                            }
                        ],
                    }
                ]
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.fireworksai import STT

    return STT(api_key="test-key", base_url=server.ws_url)


PROVIDERS = (
    Provider(
        "fireworksai",
        Whisper,
        create_stt,
        ("https://fireworks.ai/docs/api-reference/audio-streaming-transcriptions",),
        completion_unknown_reason="Legacy Fireworks checkpoint completion needs a verified server trace",
        transport_fixture=TransportFixture(
            "fireworksai",
            plugin("fireworksai.stt"),
            completion_messages=(
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
            stop_message={"checkpoint_id": "final"},
        ),
        live_constructor="livekit.plugins.fireworksai.stt:STT",
        required_env=("FIREWORKS_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
)
