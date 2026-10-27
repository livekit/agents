"""Sarvam legacy JSON audio and realtime binary audio are separate wire protocols."""

import base64

import aiohttp

from tests.stt_harness.faults import FaultCase, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Legacy(Protocol):
    async def open(self, request):
        self.configured = True
        self.sent_segments = 0

    async def receive(self, message):
        event = json_message(message)
        peer = self.connection
        if event.get("type") == "end_of_stream":
            peer.end_input()
            peer.server.final_requested.set()
        elif "audio" in event:
            await peer.accept_audio(base64.b64decode(event["audio"]["data"], validate=True))
        else:
            assert event.get("type") in ("config", "flush")

    async def transcript(self, result):
        offset = 0
        complete = []
        for segment in self.connection.server.segments or ():
            offset += len(segment.audio)
            if offset <= len(self.connection.server.received_audio):
                complete.append(segment)
        for segment in complete[self.sent_segments :]:
            self.connection.send(
                {
                    "type": "data",
                    "data": {
                        "transcript": segment.text,
                        "language_code": "en-IN",
                        "request_id": "test-session",
                    },
                }
            )
        self.sent_segments = len(complete)


class Realtime(Protocol):
    async def open(self, request):
        self.configured = True
        self.turn_open = False
        self.connection.send({"event": "session.begin", "session_id": "test-session"})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        kind = event["event"]
        if kind == "end":
            peer.end_input()
            await peer.finalize()
            peer.send(
                {"event": "session.end", "audio_duration_s": len(peer.audio) / 32000}, final=True
            )
            peer.close()
        elif kind in ("speech_end", "flush"):
            await peer.finalize()
        elif kind == "ping":
            peer.send({"event": "pong"})
        elif kind == "config.update":
            peer.send({"event": "config.updated"})
        else:
            assert kind == "speech_start"

    async def transcript(self, result):
        peer = self.connection
        if not self.turn_open:
            self.turn_open = True
            peer.send({"event": "vad.speech_start", "utterance_idx": result.index})
        peer.send(
            {
                "event": "transcript.final" if result.final else "transcript.partial",
                "utterance_idx": result.index,
                "text": result.text,
                "language": "en-IN",
            },
            final=result.final,
        )
        if result.final:
            peer.send({"event": "vad.speech_end", "utterance_idx": result.index}, final=True)
            self.turn_open = False


def legacy(server, monkeypatch):
    from livekit.plugins.sarvam import stt as module

    monkeypatch.setattr(module, "_get_urls_for_model", lambda model: (server.url, server.ws_url))
    return module.STT(api_key="test-key", language="en-IN", sample_rate=16000)


def realtime(server, monkeypatch):
    from livekit.plugins.sarvam.stt_streaming import STTRealtime

    return STTRealtime(api_key="test-key", language="en-IN", base_url=server.ws_url)


PROVIDERS = (
    Provider(
        "sarvam",
        Legacy,
        legacy,
        ("https://docs.sarvam.ai/api/stt/stt-ws",),
        completion_unknown="Legacy Sarvam EOF completion needs a verified wire specification",
        attempts_after_audio=1,
        fault_case=FaultCase(
            "sarvam",
            plugin("sarvam.stt"),
            completion=(
                {
                    "type": "data",
                    "data": {"transcript": "words", "language_code": "en-IN", "request_id": "test"},
                },
            ),
            stop={"type": "end_of_stream"},
        ),
        live_constructor="livekit.plugins.sarvam.stt:STT",
        required_env=("SARVAM_API_KEY",),
        live_options={"language": "en-IN"},
        live_env={},
    ),
    Provider(
        "sarvam_realtime",
        Realtime,
        realtime,
        ("https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/realtime-streaming",),
        attempts_after_audio=1,
        fault_case=FaultCase(
            "sarvam_realtime",
            plugin("sarvam.stt_streaming"),
            "RealtimeSpeechStream",
            "STTRealtime",
            completion=(
                {"event": "vad.speech_start", "utterance_idx": 0},
                {
                    "event": "transcript.final",
                    "utterance_idx": 0,
                    "text": "words",
                    "language": "en-IN",
                },
                {"event": "vad.speech_end", "utterance_idx": 0},
                {"event": "session.end"},
            ),
            stop={"event": "end"},
        ),
        live_constructor="livekit.plugins.sarvam.stt_streaming:STTRealtime",
        required_env=("SARVAM_API_KEY",),
        live_options={"language": "en-IN"},
        live_env={},
    ),
)
