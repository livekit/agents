"""Palabra STT: binary audio, empty binary EOF, final transcripts, stream_finished."""

from dataclasses import replace

import aiohttp

from tests.stt_harness.protocol import Protocol, Provider


class Palabra(Protocol):
    async def open(self, request):
        assert request.query["token"]
        assert request.query["sample_rate"] == "16000"
        self.configured = True

    async def receive(self, message):
        assert message.type == aiohttp.WSMsgType.BINARY
        peer = self.connection
        if message.data:
            await peer.accept_audio(message.data)
        else:
            peer.end_input()
            await peer.finalize()
            peer.send({"message_type": "stream_finished"}, final=True)
            peer.close()

    async def send_transcript(self, result):
        self.connection.send(
            {
                "message_type": "transcription",
                "is_eos": result.final,
                "transcription_id": f"utterance-{result.index}",
                "language": "en",
                "segment": {
                    "text": result.text,
                    "start_time": result.start,
                    "end_time": result.end,
                },
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from palabra_ai import client

    from livekit.plugins.palabra import STT

    monkeypatch.setitem(client.REGIONS, "eu", replace(client.REGIONS["eu"], stt=server.ws_url))
    return STT(api_key="test-key", region="eu", sample_rate=16000)


PROVIDERS = (
    Provider(
        "palabra",
        Palabra,
        create_stt,
        ("https://docs.palabra.ai/docs/streaming_api/realtime_stt",),
        uses_websocket_monitor=False,
        completion_on_close=False,
    ),
)
