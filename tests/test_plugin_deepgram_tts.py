from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from types import SimpleNamespace

import aiohttp
import pytest

pytestmark = pytest.mark.plugin("deepgram")


class _FakeMsg:
    def __init__(self, type: aiohttp.WSMsgType, data=None, extra=None) -> None:
        self.type = type
        self.data = data
        self.extra = extra

    @classmethod
    def text(cls, payload: dict) -> _FakeMsg:
        return cls(aiohttp.WSMsgType.TEXT, json.dumps(payload))

    @classmethod
    def binary(cls, data: bytes) -> _FakeMsg:
        return cls(aiohttp.WSMsgType.BINARY, data)


class _FakeWS:
    """Minimal aiohttp.ClientWebSocketResponse stand-in for the receive loop."""

    def __init__(self, incoming: list[_FakeMsg]) -> None:
        self._incoming: deque[_FakeMsg] = deque(incoming)
        self.sent: list[str] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float | None = None) -> _FakeMsg:
        if self._incoming:
            return self._incoming.popleft()
        # Nothing left to hand out: emulate a closed socket so a runaway loop
        # surfaces instead of hanging.
        return _FakeMsg(aiohttp.WSMsgType.CLOSED)

    @property
    def unconsumed(self) -> list[_FakeMsg]:
        return list(self._incoming)


class _FakePool:
    def __init__(self, ws: _FakeWS) -> None:
        self._ws = ws
        self.last_acquire_time = 0.0
        self.last_connection_reused = False

    def connection(self, timeout: float):  # noqa: ANN201
        ws = self._ws

        class _Ctx:
            async def __aenter__(self_):  # noqa: N805
                return ws

            async def __aexit__(self_, *exc):  # noqa: N805
                return False

        return _Ctx()


class _FakeEmitter:
    def __init__(self) -> None:
        self.segments_started = 0
        self.segments_ended = 0
        self.pushed: list[bytes] = []

    def start_segment(self, *, segment_id: str) -> None:
        self.segments_started += 1

    def end_segment(self) -> None:
        self.segments_ended += 1

    def push(self, data: bytes) -> None:
        self.pushed.append(data)


class _FakeWordStream:
    def __init__(self, words: list[str]) -> None:
        self._words = deque(words)

    def __aiter__(self) -> _FakeWordStream:
        return self

    async def __anext__(self):  # noqa: ANN201
        if not self._words:
            raise StopAsyncIteration
        return SimpleNamespace(token=self._words.popleft())


def _make_stream(ws: _FakeWS):  # noqa: ANN201
    # Exercise SynthesizeStreamv2._run_ws without constructing the full stream (whose
    # base __init__ wires up channels and telemetry we don't need here).
    from livekit.plugins.deepgram.tts_v2 import SynthesizeStreamv2

    stream = SimpleNamespace(
        _tts=SimpleNamespace(_pool=_FakePool(ws)),
        _conn_options=SimpleNamespace(timeout=5.0),
        _FLUSH_MSG=SynthesizeStreamv2._FLUSH_MSG,
        _mark_started=lambda: None,
    )
    stream._run_ws = SynthesizeStreamv2._run_ws.__get__(stream)
    return stream


# --- update_options / pool invalidation (DG-4b) --------------------------------------


async def test_update_options_invalidates_pool_on_change():
    from livekit.plugins.deepgram import TTSv2

    tts = TTSv2(api_key="test-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]

    tts.update_options(sample_rate=48000)

    assert calls == [True]
    assert tts._opts.sample_rate == 48000
    # base-class property is kept in sync so downstream sizing stays correct
    assert tts.sample_rate == 48000


async def test_update_options_noop_leaves_pool_alone():
    from livekit.plugins.deepgram import TTSv2

    tts = TTSv2(api_key="test-key")
    calls: list[bool] = []
    tts._pool.invalidate = lambda: calls.append(True)  # type: ignore[method-assign]

    tts.update_options()

    assert calls == []


# --- receive loop (DG-4a, DG-6) ------------------------------------------------------


async def test_recv_ends_segment_on_speech_metadata_and_leaves_flushed():
    ws = _FakeWS(
        [
            _FakeMsg.text({"type": "SpeechStarted"}),
            _FakeMsg.binary(b"\x00\x01\x02\x03"),
            _FakeMsg.text({"type": "SpeechMetadata"}),
            # server emits Flushed after SpeechMetadata; the loop must not depend on it
            _FakeMsg.text({"type": "Flushed"}),
        ]
    )
    stream = _make_stream(ws)
    emitter = _FakeEmitter()

    await asyncio.wait_for(
        stream._run_ws(_FakeWordStream(["hello", "world"]), emitter), timeout=5.0
    )

    assert emitter.segments_started == 1
    assert emitter.segments_ended == 1
    assert emitter.pushed == [b"\x00\x01\x02\x03"]
    # SpeechMetadata is the authoritative end-of-turn marker; the trailing Flushed is
    # intentionally left in the buffer (absorbed as a no-op on the next pooled reuse).
    remaining = ws.unconsumed
    assert len(remaining) == 1
    assert json.loads(remaining[0].data)["type"] == "Flushed"


async def test_flushed_is_a_noop_before_speech_metadata():
    # A Flushed seen mid-turn must not end the segment early.
    ws = _FakeWS(
        [
            _FakeMsg.text({"type": "Flushed"}),
            _FakeMsg.binary(b"\xaa\xbb"),
            _FakeMsg.text({"type": "SpeechMetadata"}),
        ]
    )
    stream = _make_stream(ws)
    emitter = _FakeEmitter()

    await asyncio.wait_for(stream._run_ws(_FakeWordStream(["hi"]), emitter), timeout=5.0)

    assert emitter.segments_ended == 1
    assert emitter.pushed == [b"\xaa\xbb"]


# --- warning logging (DG-4c, DG-5) ---------------------------------------------------


async def test_warning_logs_description_and_code(caplog):
    ws = _FakeWS(
        [
            _FakeMsg.text(
                {
                    "type": "Warning",
                    "description": "No active speech detected",
                    "code": "NO_ACTIVE_SPEECH",
                }
            ),
            _FakeMsg.text({"type": "SpeechMetadata"}),
        ]
    )
    stream = _make_stream(ws)
    emitter = _FakeEmitter()

    with caplog.at_level(logging.WARNING, logger="livekit.plugins.deepgram"):
        await asyncio.wait_for(stream._run_ws(_FakeWordStream(["hi"]), emitter), timeout=5.0)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("No active speech detected" in m and "NO_ACTIVE_SPEECH" in m for m in warnings)


# --- encoding / mime type (DG-2, DG-3) -----------------------------------------------


def test_encoding_to_mimetype_maps_decodable_encodings():
    from livekit.plugins.deepgram.tts_v2 import _encoding_to_mimetype

    assert _encoding_to_mimetype("linear16") == "audio/pcm"
    assert _encoding_to_mimetype("mp3") == "audio/mp3"
    assert _encoding_to_mimetype("opus") == "audio/opus"
    assert _encoding_to_mimetype("flac") == "audio/flac"
    assert _encoding_to_mimetype("aac") == "audio/aac"


def test_encoding_to_mimetype_rejects_unsupported():
    from livekit.plugins.deepgram.tts_v2 import _encoding_to_mimetype

    # mulaw/alaw aren't playable through the pipeline yet, so they're not mapped
    for enc in ("mulaw", "alaw", "bogus"):
        with pytest.raises(ValueError):
            _encoding_to_mimetype(enc)


async def test_stream_run_rejects_non_linear16_encoding():
    # The streaming path validates encoding up front (before any connection), so a
    # compressed encoding fails fast with a clear message rather than at connect time.
    from livekit.plugins.deepgram.tts_v2 import SynthesizeStreamv2

    stream = SimpleNamespace(_opts=SimpleNamespace(encoding="mp3"))
    stream._run = SynthesizeStreamv2._run.__get__(stream)

    with pytest.raises(ValueError, match="linear16"):
        await stream._run(_FakeEmitter())  # type: ignore[arg-type]
