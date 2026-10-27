"""Scribe v2: JSON audio chunks; a commit yields correlated stable transcript copies."""

import base64

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Scribe(Protocol):
    async def open(self, request):
        self.configured = True
        self.timestamps = (
            request.query.get("include_timestamps") == "true"
            or request.query.get("include_language_detection") == "true"
        )
        self.connection.send({"message_type": "session_started", "session_id": "test-session"})

    async def receive(self, message):
        event = json_message(message)
        assert event["message_type"] == "input_audio_chunk"
        data = base64.b64decode(event["audio_base_64"], validate=True)
        if data:
            await self.connection.accept_audio(data)
        if event.get("commit"):
            await self.connection.finalize()

    async def send_transcript(self, result):
        peer = self.connection
        if not result.final:
            peer.send({"message_type": "partial_transcript", "text": result.text})
            return
        peer.send({"message_type": "committed_transcript", "text": result.text}, final=True)
        if self.timestamps:
            peer.send(
                {
                    "message_type": "committed_transcript_with_timestamps",
                    "text": result.text,
                    "words": [{"text": result.text, "start": result.start, "end": result.end}],
                },
                final=True,
            )


def create_stt(server, monkeypatch):
    from livekit.plugins.elevenlabs import STT

    return STT(
        api_key="test-key", base_url=server.url + "/v1", use_realtime=True, sample_rate=16000
    )


PROVIDERS = (
    Provider(
        "elevenlabs",
        Scribe,
        create_stt,
        ("https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime",),
        transport_fixture=TransportFixture(
            "elevenlabs",
            plugin("elevenlabs.stt"),
            options={"use_realtime": True},
            completion_messages=(
                {
                    "message_type": "committed_transcript_with_timestamps",
                    "text": "words",
                    "words": [{"text": "words", "start": 0, "end": 0.1}],
                },
            ),
            stop_message={"message_type": "input_audio_chunk", "commit": True},
        ),
        live_constructor="livekit.plugins.elevenlabs.stt:STT",
        required_env=("ELEVEN_API_KEY",),
        live_options={"use_realtime": True},
        live_options_from_env={},
        completion_on_close=False,
    ),
)
