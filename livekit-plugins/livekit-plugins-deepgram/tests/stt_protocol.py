"""Deepgram Listen v1/v2 wire contracts.

https://developers.deepgram.com/reference/speech-to-text/listen-streaming
https://developers.deepgram.com/reference/speech-to-text/listen-streaming-v-2
Finalize keeps the session open; CloseStream flushes results and closes it.
"""

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Listen(Protocol):
    async def open(self, request):
        assert request.headers["Authorization"].startswith("Token ")
        assert request.query["encoding"] == "linear16"
        assert request.query["sample_rate"] == "16000"
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            assert message.data, "Deepgram does not use empty audio to finish"
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        kind = event["type"]
        if kind == "Finalize":
            await peer.finalize()
        elif kind == "CloseStream":
            peer.end_input()
            await peer.finalize()
            peer.send({"type": "Metadata", "request_id": "test-session", "duration": 1})
            peer.close()
        else:
            assert kind in ("KeepAlive", "Configure"), f"unknown Deepgram command: {kind}"

    async def send_transcript(self, result):
        self.connection.send(
            {
                "type": "Results",
                "metadata": {"request_id": "test-session"},
                "is_final": result.final,
                "speech_final": result.final,
                "start": result.start,
                "duration": result.end - result.start,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": result.text,
                            "confidence": 1,
                            "words": [
                                {
                                    "word": result.text,
                                    "start": result.start,
                                    "end": result.end,
                                    "confidence": 1,
                                }
                            ],
                        }
                    ]
                },
            },
            final=result.final,
        )


class Flux(Listen):
    async def send_transcript(self, result):
        if not getattr(self, "turn_open", False):
            self.turn_open = True
            self.connection.send(
                {
                    "type": "TurnInfo",
                    "event": "StartOfTurn",
                    "turn_index": result.index,
                    "transcript": "",
                    "words": [],
                }
            )
        self.connection.send(
            {
                "type": "TurnInfo",
                "event": "EndOfTurn" if result.final else "Update",
                "turn_index": result.index,
                "transcript": result.text,
                "audio_window_start": result.start,
                "audio_window_end": result.end,
                "words": [
                    {"word": result.text, "start": result.start, "end": result.end, "confidence": 1}
                ],
            },
            final=result.final,
        )
        if result.final:
            self.turn_open = False


def create_stt(server, monkeypatch):
    from livekit.plugins.deepgram import STT

    return STT(api_key="test-key", base_url=server.ws_url, sample_rate=16000)


def create_flux_stt(server, monkeypatch):
    from livekit.plugins.deepgram import STTv2

    return STTv2(api_key="test-key", base_url=server.ws_url, sample_rate=16000)


PROVIDERS = (
    Provider(
        "deepgram",
        Listen,
        create_stt,
        (__doc__.splitlines()[2],),
        transport_fixture=TransportFixture(
            "deepgram",
            plugin("deepgram.stt"),
            completion_messages=(
                {
                    "type": "Results",
                    "metadata": {"request_id": "test"},
                    "is_final": True,
                    "speech_final": True,
                    "start": 0,
                    "duration": 0.1,
                    "channel": {
                        "alternatives": [
                            {
                                "transcript": "words",
                                "confidence": 1,
                                "words": [
                                    {"word": "words", "start": 0, "end": 0.1, "confidence": 1.0}
                                ],
                            }
                        ]
                    },
                },
            ),
            stop_message={"type": "CloseStream"},
        ),
        live_constructor="livekit.plugins.deepgram.stt:STT",
        required_env=("DEEPGRAM_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
    Provider(
        "deepgram_v2",
        Flux,
        create_flux_stt,
        (__doc__.splitlines()[3],),
        transport_fixture=TransportFixture(
            "deepgram_v2",
            plugin("deepgram.stt_v2"),
            "SpeechStreamv2",
            "STTv2",
            completion_messages=(
                {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "", "words": []},
                {
                    "type": "TurnInfo",
                    "event": "EndOfTurn",
                    "transcript": "words",
                    "words": [{"word": "words", "start": 0, "end": 0.1, "confidence": 1.0}],
                },
            ),
            stop_message={"type": "CloseStream"},
        ),
        live_constructor="livekit.plugins.deepgram.stt_v2:STTv2",
        required_env=("DEEPGRAM_API_KEY",),
        live_options={},
        live_options_from_env={},
    ),
)
