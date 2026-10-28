"""HTTP session creation, then audio, stop_recording, ACK, finals, end_session."""

import base64

import aiohttp
from aiohttp import web

from tests.stt_harness.fixtures import TransportFixture, _gladia_on_send, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


async def bootstrap(server, request):
    assert request.method == "POST" and request.path == "/v2/live"
    config = await request.json()
    assert config["sample_rate"] == 16000
    return web.json_response({"id": "test-session", "url": server.ws_url + "/live"})


class Gladia(Protocol):
    async def open(self, request):
        self.configured = True
        self.connection.send({"type": "start_session", "session_id": "test-session"})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        if event["type"] == "audio_chunk":
            await peer.accept_audio(base64.b64decode(event["data"]["chunk"], validate=True))
        elif event["type"] == "stop_recording":
            # stop_recording ends one Gladia session; the public stream may continue.
            peer.end_input(require_all=False)
            await peer.acknowledge({"type": "stop_recording", "acknowledged": True})
            await peer.finalize()
            peer.send({"type": "end_session", "session_id": "test-session"}, final=True)
            peer.close()
        else:
            raise AssertionError(f"unknown Gladia command: {event}")

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "transcript",
                "data": {
                    "is_final": result.final,
                    "utterance": {
                        "text": result.text,
                        "start": result.start,
                        "end": result.end,
                        "language": "en",
                    },
                },
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.gladia import STT

    return STT(api_key="test-key", base_url=server.url + "/v2/live", sample_rate=16000)


PROVIDERS = (
    Provider(
        "gladia",
        Gladia,
        create_stt,
        (
            "https://support.gladia.io/article/how-to-handle-the-websocket-connection",
            "https://docs.gladia.io/api-reference/v2/live/callback/end-session",
        ),
        bootstrap=bootstrap,
        transport_fixture=TransportFixture(
            "gladia",
            plugin("gladia.stt"),
            on_send=_gladia_on_send,
            completion_messages=(
                {"type": "transcript", "data": {"is_final": True, "utterance": {"text": "words"}}},
                {"type": "end_session"},
            ),
            stop_message={"type": "stop_recording"},
        ),
        live_constructor="livekit.plugins.gladia.stt:STT",
        required_env=("GLADIA_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
)
