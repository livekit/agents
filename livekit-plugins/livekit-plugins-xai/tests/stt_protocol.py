"""Wait for transcript.created; audio.done produces transcript.done and closure."""

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class XAI(Protocol):
    async def open(self, request):
        self.configured = True
        self.connection.send({"type": "transcript.created", "id": "test-session"})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            event = json_message(message)
            if event["type"] == "audio.done":
                peer.end_input()
                await peer.finalize()
                peer.send({"type": "transcript.done", "text": ""}, final=True)
                peer.close()
            else:
                assert event["type"] in ("finalize", "Finalize")
                await peer.finalize()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "transcript.partial",
                "text": result.text,
                "is_final": result.final,
                "speech_final": result.final,
                "start": result.start,
                "duration": result.end - result.start,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.xai import stt as module

    monkeypatch.setattr(module, "XAI_WEBSOCKET_URL", server.ws_url)
    return module.STT(api_key="test-key")


PROVIDERS = (
    Provider(
        "xai",
        XAI,
        create_stt,
        ("https://docs.x.ai/developers/rest-api-reference/inference/speech-to-text",),
        transport_fixture=TransportFixture(
            "xai",
            plugin("xai.stt"),
            initial_messages=({"type": "transcript.created"},),
            completion_messages=({"type": "transcript.done", "text": "words"},),
            stop_message={"type": "audio.done"},
        ),
        live_constructor="livekit.plugins.xai.stt:STT",
        required_env=("XAI_API_KEY",),
        live_options={},
        live_options_from_env={},
        completion_on_close=False,
    ),
)
