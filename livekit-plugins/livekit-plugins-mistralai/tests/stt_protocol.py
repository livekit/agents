"""Voxtral Realtime: created, update, append, text deltas, end, transcription.done."""

import asyncio
import base64
from dataclasses import replace

from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Voxtral(Protocol):
    async def open(self, request):
        self.session = {
            "request_id": "test-session",
            "model": request.query.get("model", "voxtral-mini-transcribe-realtime-2602"),
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
        }
        self.emitted = ""
        self.turn_open = False
        self.flushed = asyncio.Event()
        self.connection.send({"type": "session.created", "session": self.session})

    async def receive(self, message):
        peer = self.connection
        event = json_message(message)
        kind = event["type"]
        if kind == "session.update":
            self.configured = True
            assert not peer.audio, "session settings changed after audio started"
            self.session.update(event.get("session", {}))
            assert not (
                self.session.get("diarize") and self.session.get("turn_detection") is not None
            )
            peer.send({"type": "session.updated", "session": self.session})
        elif kind == "input_audio.append":
            # Session defaults are valid even when no session.update is needed.
            self.configured = True
            if self.session.get("turn_detection") is not None and not self.turn_open:
                peer.send({"type": "turn.start", "time_s": 0})
                self.turn_open = True
            await peer.accept_audio(base64.b64decode(event["audio"], validate=True))
        elif kind == "input_audio.end":
            peer.end_input()
            if peer.final_offset < len(peer.audio):
                await peer.finalize()
            else:
                peer.send(self.done_message(""), final=True)
        else:
            assert kind == "input_audio.flush"
            await peer.finalize()
            self.flushed.set()

    async def finish_turn(self):
        await asyncio.wait_for(self.flushed.wait(), 1)
        self.flushed.clear()

    def done_message(self, text, result=None):
        return {
            "type": "transcription.done",
            "model": self.session["model"],
            "text": text,
            "language": "en",
            "segments": [
                {"text": text, "start": result.start, "end": result.end, "speaker_id": "speaker_1"}
            ]
            if result is not None and self.session.get("diarize")
            else [],
            "usage": {"prompt_audio_seconds": 1, "prompt_tokens": 1, "completion_tokens": 1},
        }

    async def send_transcript(self, result):
        if result.final:
            self.connection.send(self.done_message(result.text, result), final=True)
            self.emitted = ""
            self.turn_open = False
        else:
            assert result.text.startswith(self.emitted)
            delta = result.text[len(self.emitted) :]
            if delta:
                message = {"type": "transcription.text.delta", "text": delta}
                if self.session.get("diarize"):
                    message.update(
                        type="transcription.segment",
                        speaker_id="speaker_1",
                        start=result.start,
                        end=result.end,
                    )
                self.connection.send(message)
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
        uses_websocket_monitor=False,
        pools_connections=True,
        supports_prewarm=True,
        completion_on_close=False,
        live_constructor="livekit.plugins.mistralai.stt:STT",
        required_env=("MISTRAL_API_KEY",),
        live_options={"model": "voxtral-mini-transcribe-realtime-2602"},
    ),
)


def create_voxtral_3_stt(server, monkeypatch, **kwargs):
    from mistralai.client import Mistral

    from livekit.plugins.mistralai import STT

    return STT(
        api_key="test-key",
        model="voxtral-mini-transcribe-realtime-3",
        client=Mistral(api_key="test-key", server_url=server.url),
        **kwargs,
    )


def create_voxtral_3_diarized_stt(server, monkeypatch):
    return create_voxtral_3_stt(server, monkeypatch, diarize=True)


class ServerTurns(Voxtral):
    async def finish_turn(self):
        self.connection.send({"type": "turn.end", "time_s": len(self.connection.audio) / 32000})
        await super().finish_turn()


def create_voxtral_3_turn_detection_stt(server, monkeypatch):
    return create_voxtral_3_stt(server, monkeypatch, turn_detection={})


PROVIDERS += (
    replace(
        PROVIDERS[0],
        name="mistralai_voxtral3",
        create_stt=create_voxtral_3_stt,
        live_options={"model": "voxtral-mini-transcribe-realtime-3"},
    ),
    replace(
        PROVIDERS[0],
        name="mistralai_voxtral3_diarized",
        create_stt=create_voxtral_3_diarized_stt,
        live_options={"model": "voxtral-mini-transcribe-realtime-3", "diarize": True},
    ),
    replace(
        PROVIDERS[0],
        name="mistralai_voxtral3_turn_detection",
        protocol=ServerTurns,
        create_stt=create_voxtral_3_turn_detection_stt,
        sources=("https://docs.mistral.ai/studio/audio/speech-to-text/turn-detection",),
        live_options={"model": "voxtral-mini-transcribe-realtime-3", "turn_detection": {}},
    ),
)
