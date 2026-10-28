"""Nabrah: config, ready, binary PCM, transcript events.

Public documentation does not establish a terminal response to the client's eof.
"""

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Nabrah(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            config = json_message(message)
            assert config["api_key"]
            self.configured = True
            peer.send({"type": "ready"})
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"type": "eof"}
            peer.end_input()
            peer.server.final_requested.set()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "transcript",
                "text": result.text + (" <eot>" if result.final else ""),
                "is_final": result.final,
                "audio_processed": result.end,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.nabrah import STT

    return STT(api_key="test-key", base_url=server.ws_url)


PROVIDERS = (
    Provider(
        "nabrah",
        Nabrah,
        create_stt,
        ("https://www.nabrah.ai/en/docs/api-reference/speech-to-text/stream-speech-to-text",),
        completion_unknown_reason="Nabrah's public specification does not define EOF completion",
        expected_attempts_after_audio=1,
        transport_fixture=TransportFixture(
            "nabrah",
            plugin("nabrah.stt"),
            initial_messages=({"type": "ready"},),
            completion_messages=(
                {
                    "type": "transcript",
                    "text": "words <eot>",
                    "is_final": True,
                    "audio_processed": 0.2,
                },
            ),
            stop_message={"type": "eof"},
        ),
        live_constructor="livekit.plugins.nabrah.stt:STT",
        required_env=("NABRAH_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
)
