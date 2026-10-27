"""Soniox: configuration, binary audio, empty TEXT EOF, finished, close.

https://soniox.com/docs/api-reference/stt/websocket-api
https://soniox.com/docs/stt/rt/manual-finalization
An empty binary message is audio; it is not an end-of-input request.
"""

import aiohttp

from tests.stt_harness.faults import TransportFixture, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Soniox(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            config = json_message(message)
            assert config["audio_format"] == "pcm_s16le"
            assert config["sample_rate"] == 16000
            self.configured = True
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        elif message.data == "":
            peer.end_input()
            await peer.finalize()
            peer.send({"tokens": [], "finished": True}, final=True)
            peer.close()
        else:
            event = json_message(message)
            if event["type"] == "finalize":
                await peer.finalize()
                peer.send({"tokens": [{"text": "<fin>", "is_final": True}]}, final=True)
            else:
                assert event["type"] == "keepalive"

    async def send_transcript(self, result):
        tokens = [
            {
                "text": result.text,
                "is_final": result.final,
                "language": "en",
                "start_ms": result.start * 1000,
                "end_ms": result.end * 1000,
                "confidence": 1,
            }
        ]
        if result.final:
            tokens.append({"text": "<end>", "is_final": True})
        self.connection.send(
            {"tokens": tokens, "total_audio_proc_ms": result.end * 1000}, final=result.final
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.soniox import STT, STTOptions

    return STT(api_key="test-key", base_url=server.ws_url, params=STTOptions(sample_rate=16000))


PROVIDERS = (
    Provider(
        "soniox",
        Soniox,
        create_stt,
        (__doc__.splitlines()[2],),
        transport_fixture=TransportFixture(
            "soniox",
            plugin("soniox.stt"),
            completion_messages=(
                {
                    "tokens": [
                        {
                            "word": "words",
                            "start": 0,
                            "end": 0.1,
                            "confidence": 1.0,
                            "text": "words",
                            "is_final": True,
                            "language": "en",
                        },
                        {"text": "<end>", "is_final": True},
                    ],
                    "total_audio_proc_ms": 200,
                },
                {"tokens": [], "finished": True},
            ),
            stop_message="",
        ),
        live_constructor="livekit.plugins.soniox.stt:STT",
        required_env=("SONIOX_API_KEY",),
        live_options={},
        live_options_from_env={},
        completion_on_close=False,
    ),
)
