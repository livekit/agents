"""endTurn keeps a stream open; closeStream flushes audio and produces usage."""

import base64

from tests.stt_harness.faults import FaultCase, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Inworld(Protocol):
    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        if not self.configured:
            assert event["transcribeConfig"]["sampleRateHertz"] == 16000
            self.configured = True
        elif "audioChunk" in event:
            await peer.accept_audio(base64.b64decode(event["audioChunk"]["content"], validate=True))
        elif "endTurn" in event:
            await peer.finalize()
        else:
            assert event == {"closeStream": {}}
            peer.end_input()
            await peer.finalize()
            peer.send(
                {"result": {"usage": {"transcribedAudioMs": len(peer.audio) / 32}}}, final=True
            )
            peer.close()

    async def transcript(self, result):
        self.connection.send(
            {
                "result": {
                    "transcription": {
                        "transcript": result.text,
                        "isFinal": result.final,
                        "startTimeMs": result.start * 1000,
                        "endTimeMs": result.end * 1000,
                    }
                }
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.inworld import STT

    return STT(api_key="test-key", base_url=server.url, sample_rate=16000)


PROVIDERS = (
    Provider(
        "inworld",
        Inworld,
        create_stt,
        ("https://docs.inworld.ai/stt/best-practices",),
        fault_case=FaultCase(
            "inworld",
            plugin("inworld.stt"),
            completion=(
                {"result": {"transcription": {"transcript": "words", "isFinal": True}}},
                {"result": {"usage": {"transcribedAudioMs": 100}}},
            ),
            stop={"closeStream": {}},
            completion_closes_socket=False,
        ),
        live_constructor="livekit.plugins.inworld.stt:STT",
        required_env=("INWORLD_API_KEY",),
        live_options={},
        live_env={},
    ),
)
