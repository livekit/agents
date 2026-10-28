"""Realtime transcription: append, commit, correlated completion; clear is only an ACK.

https://developers.openai.com/api/docs/guides/realtime-transcription
The server stays open after commits. Item completions may arrive out of order.
"""

import base64

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Realtime(Protocol):
    def __init__(self, connection):
        super().__init__(connection)
        self.partial = ""

    async def open(self, request):
        assert request.headers["Authorization"].startswith("Bearer ")
        self.connection.send({"type": "session.created", "session": {"id": "test-session"}})

    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        kind = event["type"]
        if kind in ("session.update", "transcription_session.update"):
            self.configured = True
            peer.send({"type": "session.updated", "session": event["session"]})
        elif kind == "input_audio_buffer.append":
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif kind == "input_audio_buffer.commit":
            assert self.configured
            if peer.final_offset == len(peer.audio):
                await peer.acknowledge(
                    {
                        "type": "error",
                        "error": {
                            "code": "input_audio_buffer_commit_empty",
                            "event_id": event.get("event_id"),
                            "message": "buffer is empty",
                        },
                    }
                )
            else:
                await peer.finalize()
        elif kind == "input_audio_buffer.clear":
            # Clearing a buffer does not finish the pending transcription jobs.
            await peer.acknowledge({"type": "input_audio_buffer.cleared"})
        else:
            raise AssertionError(f"unknown realtime command: {kind}")

    async def send_transcript(self, result):
        peer = self.connection
        item_id = f"item-{result.index}"
        if result.final:
            self.partial = ""
            await peer.acknowledge(
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": item_id,
                    "previous_item_id": None,
                }
            )
            peer.send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": result.text,
                },
                final=True,
            )
        else:
            delta = result.text[len(self.partial) :]
            self.partial = result.text
            peer.send(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": delta,
                }
            )


def create_stt(server, monkeypatch, model="gpt-live-transcribe"):
    from livekit.plugins.openai import STT

    return STT(
        api_key="test-key",
        base_url=server.url + "/v1",
        use_realtime=True,
        model=model,
    )


def server_vad(server, monkeypatch):
    return create_stt(server, monkeypatch, model="gpt-4o-transcribe")


PROVIDERS = (
    Provider(
        "openai",
        Realtime,
        create_stt,
        (__doc__.splitlines()[2],),
        sample_rate=24000,
        pools_connections=True,
        transport_fixture=TransportFixture(
            "openai",
            plugin("openai.stt"),
            options={"use_realtime": True},
            accepts_http_session=False,
            completion_messages=(
                {"type": "input_audio_buffer.committed", "item_id": "test"},
                {"type": "input_audio_buffer.cleared"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "test",
                    "transcript": "words",
                },
            ),
            stop_message={"type": "input_audio_buffer.clear"},
        ),
        live_constructor="livekit.plugins.openai.stt:STT",
        required_env=("OPENAI_API_KEY",),
        live_options={"use_realtime": True},
        live_options_from_env={},
        completion_on_close=False,
    ),
    Provider(
        "openai_server_vad",
        Realtime,
        server_vad,
        (__doc__.splitlines()[2],),
        sample_rate=24000,
        pools_connections=True,
        completion_on_close=False,
    ),
)
