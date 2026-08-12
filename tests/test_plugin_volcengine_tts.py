"""Tests for the Volcengine bidirectional TTS plugin."""

from __future__ import annotations

import asyncio
import json
import struct
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _server_frame(
    event: int,
    payload: bytes = b"{}",
    *,
    identifier: str = "session-1",
    message_type: int = 0x9,
    serialization: int = 0x1,
) -> bytes:
    identifier_bytes = identifier.encode("utf-8")
    return b"".join(
        [
            bytes([0x11, (message_type << 4) | 0x4, serialization << 4, 0x00]),
            struct.pack(">i", event),
            struct.pack(">I", len(identifier_bytes)),
            identifier_bytes,
            struct.pack(">I", len(payload)),
            payload,
        ]
    )


def _error_frame(error_code: int, message: str) -> bytes:
    payload = message.encode("utf-8")
    return b"".join(
        [
            bytes([0x11, 0xF0, 0x00, 0x00]),
            struct.pack(">I", error_code),
            struct.pack(">I", len(payload)),
            payload,
        ]
    )


def _decode_client_frame(frame: bytes, *, has_session: bool) -> tuple[int, str | None, dict]:
    event = struct.unpack_from(">i", frame, 4)[0]
    offset = 8
    session_id = None
    if has_session:
        session_size = struct.unpack_from(">I", frame, offset)[0]
        offset += 4
        session_id = frame[offset : offset + session_size].decode("utf-8")
        offset += session_size
    payload_size = struct.unpack_from(">I", frame, offset)[0]
    offset += 4
    payload = json.loads(frame[offset : offset + payload_size])
    return event, session_id, payload


def test_build_start_session_frame() -> None:
    from livekit.plugins.volcengine.protocol import Event, build_client_message

    frame = build_client_message(
        Event.START_SESSION,
        {"namespace": "BidirectionalTTS"},
        session_id="session-1",
    )

    assert frame[:4] == bytes([0x11, 0x14, 0x10, 0x00])
    assert _decode_client_frame(frame, has_session=True) == (
        Event.START_SESSION,
        "session-1",
        {"namespace": "BidirectionalTTS"},
    )


def test_parse_audio_response() -> None:
    from livekit.plugins.volcengine.protocol import Event, MessageType, parse_server_message

    message = parse_server_message(
        _server_frame(
            Event.TTS_RESPONSE,
            b"\x01\x02\x03\x04",
            message_type=0xB,
            serialization=0,
        )
    )

    assert message.message_type == MessageType.AUDIO_ONLY_SERVER_RESPONSE
    assert message.event == Event.TTS_RESPONSE
    assert message.session_id == "session-1"
    assert message.payload == b"\x01\x02\x03\x04"


def test_tts_validates_required_options() -> None:
    from livekit.plugins.volcengine import TTS

    with pytest.raises(ValueError, match="API key"):
        TTS(api_key=None, voice="voice", sample_rate=24000)
    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="key", voice="voice", sample_rate=12345)
    with pytest.raises(ValueError, match="speech_rate"):
        TTS(api_key="key", voice="voice", speech_rate=101)


def test_resource_id_defaults_to_environment() -> None:
    from livekit.plugins.volcengine import TTS

    with patch.dict("os.environ", {"VOLCENGINE_TTS_RESOURCE_ID": "seed-tts-1.0"}):
        engine = TTS(api_key="key", voice="voice")
    assert engine.model == "seed-tts-1.0"


def test_session_payload_uses_pcm_and_optional_emotion() -> None:
    from livekit.plugins.volcengine.tts import _session_payload, _TTSOptions

    payload = _session_payload(
        _TTSOptions(
            api_key="key",
            voice="voice-id",
            resource_id="seed-tts-2.0",
            endpoint="wss://example.invalid",
            sample_rate=24000,
            speech_rate=10,
            loudness_rate=-5,
            emotion="happy",
            emotion_scale=3,
        )
    )

    assert payload["namespace"] == "BidirectionalTTS"
    assert payload["req_params"]["speaker"] == "voice-id"
    assert payload["req_params"]["audio_params"] == {
        "format": "pcm",
        "sample_rate": 24000,
        "speech_rate": 10,
        "loudness_rate": -5,
        "emotion": "happy",
        "emotion_scale": 3,
    }


async def test_connection_is_reused_across_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    import aiohttp

    from livekit.plugins.volcengine import TTS
    from livekit.plugins.volcengine.protocol import Event

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.closed = False
            self.close_code = None
            self.responses = [
                _server_frame(Event.CONNECTION_STARTED, identifier="connection-1"),
                _server_frame(Event.SESSION_STARTED),
                _server_frame(
                    Event.TTS_RESPONSE,
                    b"\x00\x00" * 240,
                    message_type=0xB,
                    serialization=0,
                ),
                _server_frame(
                    Event.SESSION_FINISHED,
                    json.dumps({"status_code": 20000000, "message": "ok"}).encode(),
                ),
                _server_frame(Event.SESSION_STARTED),
                _server_frame(
                    Event.TTS_RESPONSE,
                    b"\x00\x00" * 240,
                    message_type=0xB,
                    serialization=0,
                ),
                _server_frame(
                    Event.SESSION_FINISHED,
                    json.dumps({"status_code": 20000000, "message": "ok"}).encode(),
                ),
            ]

        async def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

        async def receive(self, *, timeout: float):
            return type(
                "Message", (), {"type": aiohttp.WSMsgType.BINARY, "data": self.responses.pop(0)}
            )()

        async def close(self) -> None:
            self.closed = True

    fake_ws = FakeWebSocket()

    class FakeSession:
        async def ws_connect(self, *args, **kwargs):
            return fake_ws

    monkeypatch.setattr("livekit.plugins.volcengine.tts.utils.shortuuid", lambda: "session-1")
    engine = TTS(
        api_key="key",
        voice="voice-id",
        http_session=FakeSession(),  # type: ignore[arg-type]
    )
    first_stream = engine.stream()
    first_stream.push_text("你好")
    first_stream.end_input()
    first_audio = [event async for event in first_stream]

    second_stream = engine.stream()
    second_stream.push_text("，世界")
    second_stream.end_input()
    second_audio = [event async for event in second_stream]
    await engine.aclose()

    decoded = [
        _decode_client_frame(frame, has_session=index not in {0, len(fake_ws.sent) - 1})
        for index, frame in enumerate(fake_ws.sent)
    ]
    assert [item[0] for item in decoded] == [
        Event.START_CONNECTION,
        Event.START_SESSION,
        Event.TASK_REQUEST,
        Event.FINISH_SESSION,
        Event.START_SESSION,
        Event.TASK_REQUEST,
        Event.FINISH_SESSION,
        Event.FINISH_CONNECTION,
    ]
    assert decoded[2][2]["req_params"]["text"] == "你好"
    assert decoded[5][2]["req_params"]["text"] == "，世界"
    assert first_audio
    assert second_audio


async def test_handshake_error_closes_websocket() -> None:
    import aiohttp

    from livekit.agents import APIStatusError
    from livekit.plugins.volcengine import TTS

    class FakeWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self.close_code = None

        async def send_bytes(self, data: bytes) -> None:
            pass

        async def receive(self, *, timeout: float):
            return type(
                "Message",
                (),
                {"type": aiohttp.WSMsgType.BINARY, "data": _error_frame(45000001, "invalid key")},
            )()

        async def close(self) -> None:
            self.closed = True

    fake_ws = FakeWebSocket()

    class FakeSession:
        async def ws_connect(self, *args, **kwargs):
            return fake_ws

    engine = TTS(
        api_key="bad-key",
        voice="voice-id",
        http_session=FakeSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(APIStatusError, match="invalid key"):
        await engine._connect_ws(timeout=1)

    assert fake_ws.closed is True


async def test_stream_receives_audio_while_text_is_still_arriving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiohttp

    from livekit.agents import APIConnectOptions
    from livekit.agents.utils import aio
    from livekit.plugins.volcengine import TTS
    from livekit.plugins.volcengine.protocol import Event
    from livekit.plugins.volcengine.tts import SynthesizeStream

    class FakeWebSocket:
        def __init__(self) -> None:
            self.text_sent = asyncio.Event()
            self.audio_received = asyncio.Event()
            self.responses = [
                _server_frame(Event.SESSION_STARTED),
                _server_frame(
                    Event.TTS_RESPONSE,
                    b"\x00\x00" * 240,
                    message_type=0xB,
                    serialization=0,
                ),
                _server_frame(
                    Event.SESSION_FINISHED,
                    json.dumps({"status_code": 20000000, "message": "ok"}).encode(),
                ),
            ]

        async def send_bytes(self, data: bytes) -> None:
            event, _, _ = _decode_client_frame(data, has_session=True)
            if event == Event.TASK_REQUEST:
                self.text_sent.set()

        async def receive(self, *, timeout: float):
            frame = self.responses.pop(0)
            if struct.unpack_from(">i", frame, 4)[0] == Event.TTS_RESPONSE:
                await self.text_sent.wait()
                self.audio_received.set()
            return type("Message", (), {"type": aiohttp.WSMsgType.BINARY, "data": frame})()

    engine = TTS(api_key="key", voice="voice-id")
    stream = SynthesizeStream(
        tts=engine,
        conn_options=APIConnectOptions(max_retry=0, timeout=1),
    )
    segment = aio.Chan[str]()

    class FakeEmitter:
        def start_segment(self, *, segment_id: str) -> None:
            pass

        def push(self, data: bytes) -> None:
            pass

        def end_segment(self) -> None:
            pass

    emitter = FakeEmitter()
    fake_ws = FakeWebSocket()

    segment.send_nowait("first token")
    session_task = asyncio.create_task(
        stream._run_session(fake_ws, segment, emitter)  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(fake_ws.audio_received.wait(), timeout=1)
        assert segment.closed is False
        segment.close()
        await session_task
    finally:
        segment.close()
        await aio.cancel_and_wait(session_task)
        await stream.aclose()
        await engine.aclose()
