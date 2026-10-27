"""Unmute Bridge: init/ready, binary PCM, transcript snapshots and finalize/close.

Current public references describe provider-native endpoints, not this bridge's EOF
ordering. Keep that gap explicit rather than infer a successful close from the client.
"""

import aiohttp

from tests.stt_harness.faults import FaultCase, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Bridge(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            event = json_message(message)
            assert event["type"] == "init"
            self.configured = True
            peer.send({"type": "ready"})
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            event = json_message(message)
            if event["type"] == "finalize":
                await peer.finalize()
            elif event["type"] == "close":
                peer.end_input()
                peer.server.final_requested.set()
            else:
                assert event["type"] == "keepalive"

    async def transcript(self, result):
        self.connection.send(
            {
                "type": "final_transcript" if result.final else "partial_transcript",
                "transcript": result.text,
                "start": result.start,
                "duration": result.end - result.start,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.slng import STT

    return STT(api_key="test-key", slng_base_url=server.url, model="deepgram/nova-3")


PROVIDERS = (
    Provider(
        "slng",
        Bridge,
        create_stt,
        ("https://docs.slng.ai/api-reference/speech-to-text/overview",),
        progress_requires_final=True,
        completion_unknown="Unmute Bridge EOF ordering needs a verified specification or trace",
        fault_case=FaultCase(
            "slng",
            plugin("slng.stt"),
            options={"model": "deepgram/nova-3"},
            greetings=({"type": "ready"},),
            completion=({"type": "final_transcript", "transcript": "words"},),
            stop={"type": "close"},
        ),
        live_constructor="livekit.plugins.slng.stt:STT",
        required_env=("SLNG_API_KEY",),
        live_options={"model": "deepgram/nova-3"},
        live_env={},
    ),
)
