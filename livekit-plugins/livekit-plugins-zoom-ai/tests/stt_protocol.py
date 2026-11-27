"""Scribe Live: session.update, binary PCM16, server VAD finals, session.close drain.

https://developers.zoom.us/docs/ai-services/scribe/live-mode/
After session.close the server sends any remaining transcription.completed, then
session.closed; the client then closes the WebSocket. Errors are {code, message, fatal}.
"""

import aiohttp

from tests.stt_harness.fixtures import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider, RequestError
from tests.stt_harness.server import json_message


class ScribeLive(Protocol):
    async def open(self, request):
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["x-zm-trackingid"]
        # Scribe's own VAD ends each turn (the default turn_detection="server").
        self.connection.server.automatic_finals = True
        self.turn_open = False
        self.connection.send({"type": "session.created", "session_id": "test-session"})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            assert message.data, "Scribe audio frames are non-empty PCM16"
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        kind = event["type"]
        if not self.configured:
            assert kind == "session.update", "session.update must come first"
            assert event["audio"]["format"] == "pcm16"
            assert event["language"]
            self.configured = True
            peer.send(
                {
                    "type": "session.updated",
                    "language": event["language"],
                    "diarization_enabled": False,
                }
            )
        elif kind == "input_audio_buffer.commit":
            await peer.finalize()
        elif kind == "session.close":
            peer.end_input()
            await peer.finalize()
            peer.send({"type": "session.closed", "reason": "client_close"}, final=True)
        else:
            raise AssertionError(f"unknown Scribe Live client event: {kind}")

    async def send_transcript(self, result):
        peer = self.connection
        item = f"item-{result.index}"
        if not self.turn_open:
            self.turn_open = True
            peer.send(
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": item,
                    "audio_start_ms": int(result.start * 1000),
                }
            )
        if not result.final:
            peer.send({"type": "transcription.delta", "item_id": item, "delta": result.text})
            return
        self.turn_open = False
        end_ms = int(result.end * 1000)
        peer.send(
            {"type": "input_audio_buffer.speech_stopped", "item_id": item, "audio_end_ms": end_ms},
            final=True,
        )
        peer.send(
            {
                "type": "transcription.completed",
                "item_id": item,
                "transcript": result.text,
                "audio_start_ms": int(result.start * 1000),
                "audio_end_ms": end_ms,
                "transcription_latency_ms": 10,
            },
            final=True,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.zoom_ai import STT

    return STT(api_key="test-key", base_url=server.url)


def _fatal_error(code, message):
    return {"type": "error", "error": {"code": code, "message": message, "fatal": True}}


PROVIDERS = (
    Provider(
        "zoom_ai",
        ScribeLive,
        create_stt,
        (__doc__.splitlines()[2],),
        request_errors=(
            RequestError("auth", http_status=401),
            # A fatal error in place of session.updated rejects the session's settings.
            RequestError(
                "rejected_settings", body=_fatal_error("invalid_config", "unsupported language")
            ),
            # A server-side end (session time limit, idle timeout) needs a new session.
            RequestError(
                "session_closed",
                body={"type": "session.closed", "reason": "max_duration"},
                retryable=True,
            ),
        ),
        transport_fixture=TransportFixture(
            "zoom_ai",
            plugin("zoom_ai.stt"),
            initial_messages=({"type": "session.created"}, {"type": "session.updated"}),
            completion_messages=(
                {
                    "type": "transcription.completed",
                    "item_id": "test",
                    "transcript": "words",
                    "audio_start_ms": 0,
                    "audio_end_ms": 100,
                },
                {"type": "session.closed", "reason": "client_close"},
            ),
            stop_message={"type": "session.close"},
        ),
        live_constructor="livekit.plugins.zoom_ai.stt:STT",
        required_env=("ZOOM_AI_API_KEY",),
        live_options={},
        live_options_from_env={},
        progress_requires_final=True,
    ),
)
