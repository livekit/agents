"""Turns: binary PCM, candidates, flush_request, final candidate and turn_end.

A flush only produces turn events while a turn is active. There is no stream EOS ACK.
"""

import aiohttp

from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Turns(Protocol):
    async def open(self, request):
        self.configured = True
        self.turn_open = False

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"type": "flush_request"}
            peer.server.final_requested.set()
            if self.turn_open:
                await peer.finalize()

    async def send_transcript(self, result):
        peer = self.connection
        if not self.turn_open:
            self.turn_open = True
            peer.send({"type": "turn_start"})
        peer.send(
            {
                "type": "turn_end_candidate",
                "text": result.text,
                "start_ms": result.start * 1000,
                "duration_ms": (result.end - result.start) * 1000,
            },
            final=result.final,
        )
        if result.final:
            peer.send({"type": "turn_end"}, final=True)
            self.turn_open = False


def create_stt(server, monkeypatch):
    from livekit.plugins.reson8 import STT

    return STT(api_key="test-key", base_url=server.url)


PROVIDERS = (
    Provider(
        "reson8",
        Turns,
        create_stt,
        ("https://docs.reson8.dev/api/speech-to-text/turns/",),
        uses_websocket_monitor=False,
        final_response_timeout=0.8,
        completion_on_close=False,
    ),
)
