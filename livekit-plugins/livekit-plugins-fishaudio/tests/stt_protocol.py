"""Fish Audio realtime ASR: MessagePack maps in binary frames, both directions.

https://docs.fish.audio/api-reference/endpoint/websocket/asr-realtime
`finalize` commits the open turn and keeps the session open. `audio_done` commits the
open turn, then the server sends `session_done` and closes with 1000.
"""

import aiohttp
import msgpack

from tests.stt_harness.protocol import Protocol, Provider, RequestError


class Realtime(Protocol):
    def __init__(self, connection):
        super().__init__(connection)
        self.turn_open = False

    async def open(self, request):
        assert request.path == "/v1/asr/realtime"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["model"]

    def emit(self, event, *, final=False):
        self.connection.send(msgpack.packb(event), final=final)

    async def receive(self, message):
        peer = self.connection
        assert message.type == aiohttp.WSMsgType.BINARY, "Fish Audio accepts binary frames only"
        event = msgpack.unpackb(message.data)
        kind = event["type"]
        if not self.configured:
            assert kind == "config", "the first message must be the config"
            assert event["audio"] == {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1}
            assert event.get("protocol_version", 1) == 1
            self.configured = True
            self.emit(
                {"type": "session_started", "session_id": "test-session", "request_id": "test"}
            )
        elif kind == "audio":
            assert isinstance(event["data"], bytes), "audio is MessagePack bin, not base64"
            await peer.accept_audio(event["data"])
        elif kind == "finalize":
            await peer.finalize()
        elif kind == "audio_done":
            peer.end_input()
            await peer.finalize()
            self.emit({"type": "session_done", "stats": {}, "request_id": "test"}, final=True)
            peer.close()
        else:
            assert kind in ("keepalive", "configure", "context_add"), f"unknown message: {kind}"

    async def send_transcript(self, result):
        if not self.turn_open:
            self.turn_open = True
            start_ms = int(result.start * 1000)
            self.emit({"type": "speech_start", "turn_index": result.index, "audio_ms": start_ms})
            self.emit({"type": "user_turn_taken", "turn_index": result.index, "audio_ms": start_ms})
        if not result.final:
            self.emit(
                {
                    "type": "draft",
                    "turn_index": result.index,
                    "text": result.text,
                    "audio_ms": int(result.end * 1000),
                }
            )
            return
        self.turn_open = False
        self.emit(
            {
                "type": "committed",
                "turn_index": result.index,
                "text": result.text,
                "words": [
                    {
                        "text": result.text,
                        "start_ms": int(result.start * 1000),
                        "end_ms": int(result.end * 1000),
                    }
                ],
                "language_code": "en",
                "audio_start_ms": int(result.start * 1000),
                "audio_end_ms": int(result.end * 1000),
                "reason": "finalize",
                "degraded": False,
                "user_turn": True,
                "turn_kind": "user",
            },
            final=True,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.fishaudio import STT

    return STT(
        api_key="test-key",
        base_url=server.url,
        sample_rate=16000,
        http_session=server.client_session(),
    )


PROVIDERS = (
    Provider(
        "fishaudio",
        Realtime,
        create_stt,
        (__doc__.splitlines()[2],),
        # Upgrade refusals and error close codes from the Errors section of the reference.
        request_errors=(
            RequestError("auth", http_status=401),
            RequestError("concurrency_limit", http_status=429, retryable=True),
            RequestError("invalid_config", close_code=4400, close_reason="invalid_config"),
        ),
        live_constructor="livekit.plugins.fishaudio:STT",
        required_env=("FISH_API_KEY",),
    ),
)
