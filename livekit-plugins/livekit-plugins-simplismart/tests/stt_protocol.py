"""Experimental streaming: binary audio in, binary UTF-8 text out; EOF is undocumented."""

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Simplismart(Protocol):
    async def receive(self, message):
        if not self.configured:
            assert "language" in json_message(message)
            self.configured = True
            return
        assert message.type == aiohttp.WSMsgType.BINARY
        await self.connection.accept_audio(message.data)

    async def send_transcript(self, result):
        if result.text:
            self.connection.send(result.text.encode(), final=result.final)


def create_stt(server, monkeypatch):
    from livekit.plugins.simplismart import STT

    original_session = aiohttp.ClientSession

    def session(*args, **kwargs):
        kwargs.setdefault("connector", aiohttp.TCPConnector(ssl=server.ssl_context))
        return original_session(*args, **kwargs)

    # stream() owns a new session; retain real networking with the local test CA.
    monkeypatch.setattr(aiohttp, "ClientSession", session)
    return STT(
        api_key="test-key",
        base_url=server.url,
        streaming=True,
    )


PROVIDERS = (
    Provider(
        "simplismart",
        Simplismart,
        create_stt,
        (
            "https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-simplismart",
        ),
        completion_unknown_reason="Experimental Simplismart streaming has no published EOF protocol",
        tls=True,
        transport_fixture=TransportFixture(
            "simplismart",
            plugin("simplismart.stt"),
            options={"streaming": True},
            completion_messages=(b"words",),
            stop_message=None,
        ),
        live_constructor="livekit.plugins.simplismart.stt:STT",
        required_env=("SIMPLISMART_API_KEY",),
        live_options={"streaming": True},
        live_options_from_env={},
    ),
)
