"""Voxtral Realtime: created, update, append, text deltas, end, transcription.done."""

import base64

from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Voxtral(Protocol):
    async def open(self, request):
        self.session = {
            "request_id": "test-session",
            "model": "voxtral-mini-transcribe-realtime-2602",
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
        }
        self.emitted = ""
        self.connection.send({"type": "session.created", "session": self.session})

    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        kind = event["type"]
        if kind == "session.update":
            self.configured = True
            self.session.update(event.get("session", {}))
            peer.send({"type": "session.updated", "session": self.session})
        elif kind == "input_audio.append":
            # Session defaults are valid even when no session.update is needed.
            self.configured = True
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif kind == "input_audio.end":
            peer.end_input()
            await peer.finalize()
        else:
            assert kind == "input_audio.flush"

    async def transcript(self, result):
        if result.final:
            self.connection.send(
                {
                    "type": "transcription.done",
                    "model": self.session["model"],
                    "text": result.text,
                    "language": "en",
                    "segments": [],
                    "usage": {
                        "prompt_audio_seconds": 1,
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                final=True,
            )
        else:
            assert result.text.startswith(self.emitted)
            delta = result.text[len(self.emitted) :]
            if delta:
                self.connection.send({"type": "transcription.text.delta", "text": delta})
            self.emitted = result.text


def create_stt(server, monkeypatch):
    from mistralai.client import Mistral

    from livekit.plugins.mistralai import STT
    from tests.stt_harness.drivers import NoSpeechVAD

    return STT(
        api_key="test-key",
        model="voxtral-mini-transcribe-realtime-2602",
        client=Mistral(api_key="test-key", server_url=server.url),
        vad=NoSpeechVAD(),
    )


PROVIDERS = (
    Provider(
        "mistralai",
        Voxtral,
        create_stt,
        ("https://docs.mistral.ai/studio/audio/speech-to-text/transcribe-in-realtime",),
        monitored=False,
        completion_on_close=False,
    ),
)
