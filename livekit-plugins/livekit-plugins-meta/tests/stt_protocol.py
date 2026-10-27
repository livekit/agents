"""Muse ASR: handshake, acknowledgement, binary 24 kHz PCM, endStream, results, 1000."""

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Muse(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            config = json_message(message)
            assert config["audioEncoding"] == "PCM_24KHZ"
            self.configured = True
            peer.send({"sessionId": "test-session"})
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
            if not peer.server.processing_stalled and not peer.server.fault_after_audio:
                peer.send({"type": "audioProgress", "audioProcessedMs": len(peer.audio) / 48})
        else:
            assert json_message(message) == {"type": "endStream"}
            peer.end_input()
            await peer.finalize()
            peer.close()

    async def send_transcript(self, result):
        peer = self.connection
        turn = f"turn-{result.index}"
        if not getattr(self, "turn_open", False):
            self.turn_open = True
            peer.send({"type": "speechStart", "turnId": turn})
        if result.final:
            peer.send({"type": "speechEnd", "turnId": turn}, final=True)
            peer.send(
                {"type": "speechComplete", "turnId": turn, "transcript": result.text}, final=True
            )
            self.turn_open = False
        else:
            peer.send(
                {
                    "type": "transcript",
                    "turnId": turn,
                    "text": result.text,
                    "transcript": result.text,
                    "final": False,
                }
            )


def create_stt(server, monkeypatch):
    from livekit.plugins.meta import STT

    return STT(api_key="test-key", url=server.ws_url, http_session=server.client_session())


PROVIDERS = (
    Provider(
        "meta",
        Muse,
        create_stt,
        ("https://dev.meta.ai/docs/speech-to-text",),
        sample_rate=24000,
        tls=True,
        expected_attempts_after_audio=1,
        transport_fixture=TransportFixture(
            "meta",
            plugin("meta.stt"),
            initial_messages=({"sessionId": "test"},),
            completion_messages=(
                {"type": "speechStart", "turnId": "test"},
                {"type": "speechEnd", "turnId": "test"},
                {"type": "speechComplete", "turnId": "test", "transcript": "words"},
            ),
            stop_message={"type": "endStream"},
        ),
        live_constructor="livekit.plugins.meta.stt:STT",
        required_env=("META_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
)
