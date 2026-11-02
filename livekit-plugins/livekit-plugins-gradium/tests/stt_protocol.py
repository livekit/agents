"""Gradium: setup/ready, JSON audio, incremental text, end_of_stream in both directions."""

import base64

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Gradium(Protocol):
    async def receive(self, message):
        event = json_message(message)
        peer = self.connection
        kind = event["type"]
        if not self.configured:
            assert kind == "setup"
            self.configured = True
            self.emitted_words = 0
            peer.send({"type": "ready", "request_id": "test-session"})
        elif kind == "audio":
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif kind == "end_of_stream":
            peer.end_input()
            await peer.finalize()
            peer.send({"type": "end_text", "stop_s": len(peer.audio) / 48000}, final=True)
            peer.send({"type": "end_of_stream"}, final=True)
            peer.close()
        elif kind == "flush":
            await peer.finalize()
            peer.send({"type": "flushed"}, final=True)
        else:
            raise AssertionError(f"unknown Gradium message: {event}")

    async def send_transcript(self, result):
        offset = 0
        words = []
        for segment in self.connection.server.segments or ():
            offset += len(segment.audio)
            if offset <= len(self.connection.server.received_audio):
                words.append(segment.text)
        for word in words[self.emitted_words :]:
            self.connection.send(
                {"type": "text", "text": word, "start_s": result.start}, final=result.final
            )
        self.emitted_words = len(words)

    async def finish_turn(self):
        await super().finish_turn()
        self.connection.send(
            {"type": "end_text", "stop_s": len(self.connection.audio) / 48000}, final=True
        )
        # Six tokens of decoder lookahead follow the first inactivity step.
        for _ in range(7):
            self.connection.send(
                {"type": "step", "vad": [{"inactivity_prob": 1.0}] * 3}, final=True
            )


def create_stt(server, monkeypatch):
    from livekit.plugins.gradium import STT

    return STT(api_key="test-key", model_endpoint=server.ws_url, vad_flush=False)


PROVIDERS = (
    Provider(
        "gradium",
        Gradium,
        create_stt,
        ("https://docs.gradium.ai/guides/websocket-lifecycle",),
        sample_rate=24000,
        transport_fixture=TransportFixture(
            "gradium",
            plugin("gradium.stt"),
            completion_messages=(
                {"type": "text", "text": "words", "start_s": 0},
                {"type": "end_text", "stop_s": 0.1},
                {"type": "end_of_stream"},
            ),
            initial_messages=({"type": "ready"},),
            stop_message={"type": "end_of_stream"},
        ),
        live_constructor="livekit.plugins.gradium.stt:STT",
        required_env=("GRADIUM_API_KEY",),
        live_options={},
        live_options_from_env={},
        completion_on_close=False,
    ),
)
