"""Tests for the Gradium TTS plugin websocket handling, against a scripted fake server."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import APIConnectOptions, APIError, APIStatusError, APITimeoutError

# hermetic: every test talks to a scripted fake websocket, so this is a unit module
pytestmark = pytest.mark.unit

# no retries: the tests assert on the error of the single attempt
CONN_OPTS = APIConnectOptions(max_retry=0, timeout=5)

SAMPLE_RATE = 48000
# 0.5s of int16 mono silence, enough to fill the emitter's 200ms frames
AUDIO_CHUNK = bytes(SAMPLE_RATE // 2 * 2)

READY = {"type": "ready", "request_id": "req", "model_name": "default", "sample_rate": SAMPLE_RATE}
AUDIO = {
    "type": "audio",
    "audio": base64.b64encode(AUDIO_CHUNK).decode(),
    "start_s": 0,
    "stop_s": 0.5,
}
EOS = {"type": "end_of_stream"}
ERROR = {"type": "error", "message": "Unknown voice 'nope'.", "code": 1011}


@dataclass
class _Msg:
    type: Any
    data: Any = None
    extra: Any = None


class FakeWebSocket:
    """Scripted server side: records what the plugin sends, replays `frames`.

    `frames` holds JSON dicts (sent as TEXT) or the string "close" (a CLOSE frame with
    `close_code`). Once exhausted, every receive() reports the socket as CLOSED.
    """

    def __init__(self, frames: list[Any], *, close_code: int | None = None) -> None:
        import aiohttp

        self._ws_type = aiohttp.WSMsgType
        self._frames = list(frames)
        self.close_code = close_code
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive(self, timeout: float | None = None) -> _Msg:
        if not self._frames:
            return _Msg(self._ws_type.CLOSED)
        frame = self._frames.pop(0)
        if frame == "close":
            return _Msg(self._ws_type.CLOSE, self.close_code, "")
        return _Msg(self._ws_type.TEXT, json.dumps(frame))

    async def close(self) -> bool:
        return True


class FakeSession:
    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws
        self.connects: list[tuple[str, dict[str, Any]]] = []

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        self.connects.append((url, kwargs))
        return self.ws


class HangingSession:
    """A server that never completes the websocket handshake."""

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _make_tts(frames: list[Any], *, close_code: int | None = None, **kwargs: Any):
    from livekit.plugins.gradium import TTS

    ws = FakeWebSocket(frames, close_code=close_code)
    session = FakeSession(ws)
    tts = TTS(api_key="test-key", http_session=session, **kwargs)  # type: ignore[arg-type]
    return tts, session, ws


def test_model_reports_model_name():
    from livekit.plugins.gradium import TTS

    assert TTS(api_key="test-key", model_name="my-model").model == "my-model"


async def test_synthesize_emits_audio_and_sends_eos():
    tts, session, ws = _make_tts([READY, AUDIO, AUDIO, EOS])

    duration = 0.0
    async for ev in tts.synthesize("hello world", conn_options=CONN_OPTS):
        duration += ev.frame.duration
    # two 0.5s chunks; the emitter may pad the tail by a few ms
    assert 1.0 <= duration <= 1.1

    assert [m["type"] for m in ws.sent] == ["setup", "text", "end_of_stream"]
    assert ws.sent[1]["text"] == "hello world"

    headers = session.connects[0][1]["headers"]
    assert headers["x-api-key"] == "test-key"
    assert headers["x-api-source"] == "livekit"


async def test_synthesize_raises_on_error_frame():
    tts, _, _ = _make_tts([READY, ERROR, "close"], close_code=1011)

    with pytest.raises(APIError, match="Unknown voice 'nope'"):
        async for _ in tts.synthesize("hello world", conn_options=CONN_OPTS):
            pass


async def test_synthesize_raises_on_close_before_eos():
    tts, _, _ = _make_tts([READY, AUDIO, "close"], close_code=1011)

    with pytest.raises(APIStatusError) as excinfo:
        async for _ in tts.synthesize("hello world", conn_options=CONN_OPTS):
            pass
    assert excinfo.value.status_code == 1011


async def test_stream_emits_audio_and_sends_words_then_eos():
    tts, session, ws = _make_tts([READY, AUDIO, AUDIO, EOS])

    duration = 0.0
    async with tts.stream(conn_options=CONN_OPTS) as stream:
        stream.push_text("hello ")
        stream.push_text("world")
        stream.end_input()
        async for ev in stream:
            duration += ev.frame.duration
    # two 0.5s chunks; the emitter may pad the tail by a few ms
    assert 1.0 <= duration <= 1.1

    assert [m["type"] for m in ws.sent] == ["setup", "text", "text", "end_of_stream"]
    assert [m["text"] for m in ws.sent[1:3]] == ["hello ", "world "]
    assert session.connects[0][1]["headers"]["x-api-source"] == "livekit"


async def test_stream_raises_on_error_frame():
    tts, _, _ = _make_tts([READY, ERROR, "close"], close_code=1011)

    with pytest.raises(APIError, match="Unknown voice 'nope'"):
        async with tts.stream(conn_options=CONN_OPTS) as stream:
            stream.push_text("hello world")
            stream.end_input()
            async for _ in stream:
                pass


async def test_stream_raises_on_close_before_eos():
    tts, _, _ = _make_tts([READY, AUDIO, "close"], close_code=1011)

    with pytest.raises(APIStatusError) as excinfo:
        async with tts.stream(conn_options=CONN_OPTS) as stream:
            stream.push_text("hello world")
            stream.end_input()
            async for _ in stream:
                pass
    assert excinfo.value.status_code == 1011


async def _setup_sent(tts, ws) -> dict[str, Any]:
    async for _ in tts.synthesize("hi", conn_options=CONN_OPTS):
        pass
    return ws.sent[0]


async def test_default_voice_id_when_no_voice_given():
    from livekit.plugins.gradium.tts import DEFAULT_VOICE_ID

    tts, _, ws = _make_tts([READY, AUDIO, EOS])
    setup = await _setup_sent(tts, ws)
    assert setup["voice_id"] == DEFAULT_VOICE_ID
    assert "voice" not in setup


async def test_voice_name_is_sent_without_voice_id():
    tts, _, ws = _make_tts([READY, AUDIO, EOS], voice="narrator")
    setup = await _setup_sent(tts, ws)
    assert setup["voice"] == "narrator"
    assert "voice_id" not in setup


def test_voice_and_voice_id_are_exclusive():
    from livekit.plugins.gradium import TTS

    with pytest.raises(ValueError, match="mutually exclusive"):
        TTS(api_key="test-key", voice="narrator", voice_id="abc")


async def test_update_options_voice_clears_voice_id():
    tts, _, ws = _make_tts([READY, AUDIO, EOS])
    tts.update_options(voice="narrator")
    setup = await _setup_sent(tts, ws)
    assert setup["voice"] == "narrator"
    assert "voice_id" not in setup


async def test_update_options_voice_id_clears_voice():
    tts, _, ws = _make_tts([READY, AUDIO, EOS], voice="narrator")
    tts.update_options(voice_id="abc")
    setup = await _setup_sent(tts, ws)
    assert setup["voice_id"] == "abc"
    assert "voice" not in setup


async def test_synthesize_times_out_on_hung_connect():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", http_session=HangingSession())  # type: ignore[arg-type]
    opts = APIConnectOptions(max_retry=0, timeout=0.2)
    with pytest.raises(APITimeoutError):
        async for _ in tts.synthesize("hi", conn_options=opts):
            pass


async def test_stream_times_out_on_hung_connect():
    from livekit.plugins.gradium import TTS

    tts = TTS(api_key="test-key", http_session=HangingSession())  # type: ignore[arg-type]
    opts = APIConnectOptions(max_retry=0, timeout=0.2)
    with pytest.raises(APITimeoutError):
        async with tts.stream(conn_options=opts) as stream:
            stream.push_text("hi")
            stream.end_input()
            async for _ in stream:
                pass
