"""LiveKit Inference STT: session creation, JSON audio, finalization, session close.

https://github.com/livekit/agents-js/blob/main/agents/src/inference/api_protos.ts
session.finalized can precede trailing transcripts. It is not session completion.
"""

import base64

from tests.stt_harness.fixtures import TransportFixture
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Inference(Protocol):
    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        kind = event["type"]
        if not self.configured:
            assert kind == "session.create"
            self.configured = True
            peer.send({"type": "session.created", "session_id": "test-session"})
        elif kind == "input_audio":
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif kind == "session.finalize":
            peer.send({"type": "session.finalized"})
            await peer.finalize()
        elif kind == "session.close":
            peer.end_input()
            peer.send({"type": "session.closed"}, final=True)
            peer.close()
        else:
            raise AssertionError(f"unknown inference command: {kind}")

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "final_transcript" if result.final else "interim_transcript",
                "transcript": result.text,
                "language": "en",
                "start": result.start,
                "duration": result.end - result.start,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.agents.inference import STT

    return STT(
        model="deepgram/nova-3",
        api_key="test-key",
        api_secret="test-secret-" * 4,
        base_url=server.url,
    )


PROVIDERS = (
    Provider(
        "inference",
        Inference,
        create_stt,
        (__doc__.splitlines()[2],),
        final_response_timeout=0.25,
        require_completion=False,
        transport_fixture=TransportFixture(
            "inference",
            "livekit.agents.inference.stt",
            options={
                "model": "deepgram/nova-3",
                "api_secret": "test-secret-" * 4,
                "base_url": "http://unused",
            },
            completion_messages=(
                {"type": "final_transcript", "transcript": "words", "language": "en"},
            ),
            stop_message={"type": "session.finalize"},
        ),
        live_constructor="livekit.agents.inference.stt:STT",
        required_env=("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"),
        live_options={"model": "deepgram/nova-3"},
        live_options_from_env={},
    ),
)
