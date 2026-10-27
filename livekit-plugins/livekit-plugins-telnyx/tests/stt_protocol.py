"""Telnyx Deepgram engine: WAV header, binary PCM, Finalize, CloseStream.

The public specification does not define CloseStream for telnyx/google/azure engines.
"""

import struct

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Telnyx(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            assert message.type == aiohttp.WSMsgType.BINARY
            header = message.data
            assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
            assert struct.unpack_from("<I", header, 24)[0] == 16000
            assert len(header) == 44
            self.configured = True
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            kind = json_message(message)["type"]
            if kind == "CloseStream":
                peer.end_input()
                await peer.finalize()
                peer.close()
            elif kind == "Finalize":
                await peer.finalize()
            else:
                assert kind == "KeepAlive"

    async def send_transcript(self, result):
        self.connection.send(
            {"transcript": result.text, "is_final": result.final, "confidence": 1},
            final=result.final,
        )


def create_stt(server, monkeypatch, engine="deepgram"):
    from livekit.plugins.telnyx import stt as module

    return module.STT(api_key="test-key", base_url=server.ws_url, transcription_engine=engine)


PROVIDERS = (
    Provider(
        "telnyx",
        Telnyx,
        create_stt,
        ("https://developers.telnyx.com/docs/voice/stt/websocket-streaming/index",),
        transport_fixture=TransportFixture(
            "telnyx",
            plugin("telnyx.stt"),
            completion_messages=({"transcript": "words", "is_final": True},),
            options={"transcription_engine": "deepgram"},
            stop_message={"type": "CloseStream"},
        ),
        live_constructor="livekit.plugins.telnyx.stt:STT",
        required_env=("TELNYX_API_KEY",),
        live_options={"transcription_engine": "deepgram"},
        live_options_from_env={},
    ),
)


def create_google_stt(server, monkeypatch):
    return create_stt(server, monkeypatch, "google")


def create_azure_stt(server, monkeypatch):
    return create_stt(server, monkeypatch, "azure")


PROVIDERS += tuple(
    Provider(
        "telnyx_" + engine,
        Telnyx,
        factory,
        PROVIDERS[0].sources,
        completion_unknown_reason="This Telnyx engine has no documented completion request or signal",
        live_constructor="livekit.plugins.telnyx.stt:STT",
        required_env=("TELNYX_API_KEY",),
        live_options={"transcription_engine": engine},
    )
    for engine, factory in (("google", create_google_stt), ("azure", create_azure_stt))
)
