"""MAI: session handshake, JSON audio, commit acknowledgement, item completion."""

import base64

from tests.stt_harness.faults import FaultCase, _microsoft_reply, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class MAI(Protocol):
    async def open(self, request):
        self.connection.send({"type": "session.created"})

    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        kind = event["type"]
        if not self.configured:
            assert kind == "session.update"
            self.configured = True
            peer.send({"type": "session.updated"})
        elif kind == "input_audio_buffer.append":
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        else:
            assert kind == "input_audio_buffer.commit"
            await peer.finalize()

    async def transcript(self, result):
        peer = self.connection
        item = f"item-{result.index}"
        if result.final:
            await peer.acknowledge({"type": "input_audio_buffer.committed", "item_id": item})
            peer.send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item,
                    "transcript": result.text,
                },
                final=True,
            )
        else:
            peer.send(
                {
                    "type": "conversation.item.input_audio_transcription.intermediate",
                    "item_id": item,
                    "intermediate": result.text,
                }
            )


def create_stt(server, monkeypatch):
    from livekit.plugins.microsoft_ai import STT

    return STT(api_key="test-key", url=server.ws_url, model="test", vad=None)


PROVIDERS = (
    Provider(
        "microsoft_ai",
        MAI,
        create_stt,
        (
            "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-realtime",
        ),
        attempts_after_audio=1,
        fault_case=FaultCase(
            "microsoft_ai",
            plugin("microsoft_ai.stt"),
            reply=_microsoft_reply,
            options={"vad": None, "url": "wss://unused", "model": "test"},
            greetings=({"type": "session.created"}, {"type": "session.updated"}),
            completion=(
                {"type": "input_audio_buffer.committed", "item_id": "test"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "test",
                    "transcript": "words",
                },
            ),
            completion_closes_socket=False,
            stop={"type": "input_audio_buffer.commit"},
        ),
        live_constructor="livekit.plugins.microsoft_ai.stt:STT",
        required_env=("MICROSOFT_AI_STT_API_KEY", "MICROSOFT_AI_STT_URL", "MICROSOFT_AI_STT_MODEL"),
        live_options={"vad": None},
        live_env={},
    ),
)
