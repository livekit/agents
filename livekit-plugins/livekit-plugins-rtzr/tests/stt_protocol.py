"""RTZR: HTTP token, lazy WebSocket connection, binary audio, text EOS, finals, close."""

import aiohttp
from aiohttp import web

from tests.stt_harness.faults import TransportFixture, _rtzr_on_send, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


async def bootstrap(server, request):
    assert request.path == "/v1/authenticate" and request.method == "POST"
    assert (await request.post())["client_id"] == "test"
    return web.json_response({"access_token": "test-token", "expire_at": 9999999999})


class RTZR(Protocol):
    async def open(self, request):
        assert request.headers["Authorization"] == "bearer test-token"
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        elif message.data == "EOS":
            peer.end_input(require_all=False)
            await peer.finalize()
            peer.close()
        else:
            assert json_message(message) == {"type": "Finalize"}
            await peer.finalize()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "seq": result.index,
                "start_at": result.start * 1000,
                "duration": (result.end - result.start) * 1000,
                "final": result.final,
                "alternatives": [{"text": result.text}],
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.rtzr import STT

    monkeypatch.setenv("RTZR_CLIENT_ID", "test")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test")
    instance = STT(sample_rate=16000, language="en")
    instance._client._api_base = server.url
    instance._client._ws_base = server.ws_url
    return instance


PROVIDERS = (
    Provider(
        "rtzr",
        RTZR,
        create_stt,
        ("https://developers.rtzr.ai/docs/en/stt-streaming/websocket/",),
        bootstrap=bootstrap,
        transport_fixture=TransportFixture(
            "rtzr",
            plugin("rtzr.stt"),
            on_send=_rtzr_on_send,
            accepts_api_key=False,
            completion_messages=(
                {
                    "alternatives": [{"text": "words"}],
                    "final": True,
                    "start_at": 0,
                    "duration": 100,
                },
            ),
            stop_message="EOS",
        ),
        live_constructor="livekit.plugins.rtzr.stt:STT",
        required_env=("RTZR_CLIENT_ID", "RTZR_CLIENT_SECRET"),
        live_options={"language": "en"},
        live_options_from_env={},
    ),
)
