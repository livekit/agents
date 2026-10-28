"""Universal Streaming v3: Begin, audio, Turn, Terminate, Termination, close."""

import json
from dataclasses import replace

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Universal(Protocol):
    async def open(self, request):
        assert request.headers["Authorization"]
        assert request.query["sample_rate"] == "16000"
        self.configured = True
        self.connection.send({"type": "Begin", "id": "test-session", "expires_at": 9999999999})

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        if event["type"] == "Terminate":
            peer.end_input()
            await peer.finalize()
            peer.send(
                {"type": "Termination", "audio_duration_seconds": 1, "session_duration_seconds": 1},
                final=True,
            )
            peer.close()
        elif event["type"] == "ForceEndpoint":
            await peer.finalize()
        else:
            assert event["type"] in ("UpdateConfiguration", "KeepAlive")

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "Turn",
                "turn_order": result.index,
                "transcript": result.text,
                "end_of_turn": result.final,
                "turn_is_formatted": result.final,
                "words": [
                    {
                        "text": result.text,
                        "start": result.start * 1000,
                        "end": result.end * 1000,
                        "confidence": 1,
                        "word_is_final": result.final,
                    }
                ],
                "end_of_turn_confidence": 1 if result.final else 0,
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.assemblyai import STT

    return STT(api_key="test-key", base_url=server.ws_url, sample_rate=16000)


PROVIDERS = (
    Provider(
        "assemblyai",
        Universal,
        create_stt,
        ("https://www.assemblyai.com/docs/streaming/message-sequence",),
        transport_fixture=TransportFixture(
            "assemblyai",
            plugin("assemblyai.stt"),
            initial_messages=({"type": "Begin", "id": "test-session", "expires_at": 1e12},),
            completion_messages=(
                {
                    "type": "Turn",
                    "transcript": "words",
                    "end_of_turn": True,
                    "turn_is_formatted": True,
                    "turn_order": 0,
                    "words": [],
                },
            ),
            stop_message={"type": "Terminate"},
        ),
        live_constructor="livekit.plugins.assemblyai.stt:STT",
        required_env=("ASSEMBLYAI_API_KEY",),
        live_options={},
        live_options_from_env={},
        completion_on_close=False,
    ),
)


LANGUAGE_CODES_36 = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko"]


class Universal36(Universal):
    async def open(self, request):
        assert request.query["speech_model"] == "universal-3-6-pro"
        assert json.loads(request.query["language_codes"]) == LANGUAGE_CODES_36
        await super().open(request)


def create_universal_36_stt(server, monkeypatch):
    from livekit.plugins.assemblyai import STT

    return STT(
        api_key="test-key",
        base_url=server.ws_url,
        sample_rate=16000,
        model="universal-3-6-pro",
        language_codes=LANGUAGE_CODES_36,
    )


PROVIDERS += (
    replace(
        PROVIDERS[0],
        name="assemblyai_universal36",
        protocol=Universal36,
        create_stt=create_universal_36_stt,
        transport_fixture=None,
        live_options={"model": "universal-3-6-pro", "language_codes": LANGUAGE_CODES_36},
    ),
)
