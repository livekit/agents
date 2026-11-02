"""Vachana: connected, fixed-size binary PCM frames, processing/transcript events.

The public SDK documents disconnect, but no acknowledged final-audio request.
"""

import aiohttp

from tests.stt_harness.protocol import Protocol, Provider, RequestError


class Vachana(Protocol):
    async def open(self, request):
        self.configured = True
        self.connection.send({"type": "connected", "sample_rate": 16000, "chunk_size": 1024})

    async def receive(self, message):
        assert message.type == aiohttp.WSMsgType.BINARY
        assert 0 < len(message.data) <= 1024
        await self.connection.accept_audio(message.data)

    async def send_transcript(self, result):
        self.connection.send({"type": "processing"})
        if result.text:
            self.connection.send(
                {"type": "transcript", "text": result.text, "segment_id": str(result.index)}
            )


def create_stt(server, monkeypatch):
    from livekit.plugins.gnani import STT

    return STT(api_key="test-key", base_url=server.url)


PROVIDERS = (
    Provider(
        "gnani",
        Vachana,
        create_stt,
        ("https://github.com/Gnani-AI-Mintlify/Gnani-Vachana",),
        request_errors=(RequestError("auth", http_status=401),),
        uses_websocket_monitor=False,
        expected_attempts_after_audio=1,
        completion_unknown_reason="Vachana does not document an acknowledged end-of-input protocol",
    ),
)
