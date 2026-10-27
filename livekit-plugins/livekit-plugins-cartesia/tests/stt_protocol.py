"""Manual STT uses text controls; Auto STT uses JSON controls and turn events."""

import aiohttp

from tests.stt_harness.faults import FaultCase, plugin
from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class Manual(Protocol):
    async def open(self, request):
        assert request.query["sample_rate"] == "16000"
        self.configured = True

    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        elif message.data == "finalize":
            await peer.finalize()
        elif message.data == "close":
            peer.end_input()
            await peer.finalize()
            peer.send({"type": "done"}, final=True)
            peer.close()
        else:
            raise AssertionError(f"unknown Cartesia command: {message.data!r}")

    async def transcript(self, result):
        self.connection.send(
            {
                "type": "transcript",
                "text": result.text,
                "is_final": result.final,
                "request_id": "test-session",
                "duration": result.end - result.start,
                "language": "en",
                "words": [],
            },
            final=result.final,
        )


class Auto(Manual):
    async def open(self, request):
        await super().open(request)
        self.turn_open = False
        self.connection.send({"type": "connected", "request_id": "test-session"})

    async def receive(self, message):
        if message.type == aiohttp.WSMsgType.BINARY:
            await self.connection.accept_audio(message.data)
            return
        event = json_message(message)
        if event["type"] == "close":
            self.connection.end_input()
            await self.connection.finalize()
            self.connection.close()
        else:
            assert event["type"] == "config"

    async def transcript(self, result):
        if not self.turn_open:
            self.turn_open = True
            self.connection.send({"type": "turn.start", "request_id": "test-session"})
        self.connection.send(
            {
                "type": "turn.end" if result.final else "turn.update",
                "transcript": result.text,
                "request_id": "test-session",
            },
            final=result.final,
        )
        if result.final:
            self.turn_open = False


def manual(server, monkeypatch):
    from livekit.plugins.cartesia import STT

    return STT(api_key="test-key", base_url=server.url, model="ink-whisper", language="en")


def automatic(server, monkeypatch):
    from livekit.plugins.cartesia import STT

    return STT(api_key="test-key", base_url=server.url, model="ink-2", language="en")


PROVIDERS = (
    Provider(
        "cartesia_legacy",
        Manual,
        manual,
        ("https://docs.cartesia.ai/api-reference/stt/websocket",),
        fault_case=FaultCase(
            "cartesia_legacy",
            plugin("cartesia._recognize_streams.legacy_recognize_stream"),
            "LegacyRecognizeStream",
            options={"model": "ink-whisper", "language": "en"},
            constructor_module=plugin("cartesia.stt"),
            completion=(
                {"type": "transcript", "text": "words", "is_final": True, "duration": 0.1},
            ),
            stop="close",
        ),
        live_constructor="livekit.plugins.cartesia.stt:STT",
        required_env=("CARTESIA_API_KEY",),
        live_options={"model": "ink-whisper", "language": "en"},
        live_env={},
    ),
    Provider(
        "cartesia",
        Auto,
        automatic,
        ("https://docs.cartesia.ai/api-reference/stt/turns/websocket",),
        fault_case=FaultCase(
            "cartesia",
            plugin("cartesia._recognize_streams.auto_finalize_recognize_stream"),
            "AutoFinalizeRecognizeStream",
            options={"model": "ink-2", "language": "en"},
            constructor_module=plugin("cartesia.stt"),
            completion=({"type": "turn.start"}, {"type": "turn.end", "transcript": "words"}),
            stop={"type": "close"},
        ),
        live_constructor="livekit.plugins.cartesia.stt:STT",
        required_env=("CARTESIA_API_KEY",),
        live_options={"model": "ink-2", "language": "en"},
        live_env={},
    ),
)
