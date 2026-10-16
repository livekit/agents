"""Tests for Gradium STT plugin configuration and behavior."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError
from livekit.agents.stt import SpeechEventType

# hermetic: every test talks to a fake websocket, so this is a unit module
pytestmark = pytest.mark.unit

# no retries: the tests assert on the behaviour of the single attempt
CONN_OPTS = APIConnectOptions(max_retry=0, timeout=5)
SAMPLE_RATE = 24000
VAD_HORIZONS = (0.5, 1.0, 2.0, 3.0)

READY = {
    "type": "ready",
    "request_id": "req",
    "model_name": "default",
    "sample_rate": SAMPLE_RATE,
    "frame_size": 1920,
    "delay_in_frames": 3,
}
EOS = {"type": "end_of_stream"}
SILENT = (0.99,) * len(VAD_HORIZONS)
ACTIVE = (0.01,) * len(VAD_HORIZONS)


def _text(text: str, start_s: float = 0.0) -> dict[str, Any]:
    return {"type": "text", "text": text, "start_s": start_s}


def _step(idx: int, probs: tuple[float, ...]) -> dict[str, Any]:
    return {
        "type": "step",
        "step_idx": idx,
        "step_duration_s": 0.08,
        "total_duration_s": idx * 0.08,
        "vad": [
            {"horizon_s": h, "inactivity_prob": p}
            for h, p in zip(VAD_HORIZONS, probs, strict=False)
        ],
    }


@dataclass
class _Msg:
    type: Any
    data: Any = None
    extra: Any = None


class FakeWebSocket:
    """Scripted server side.

    `frames` are replayed in order (JSON dicts as TEXT, the string "close" as a CLOSE frame
    with `close_code`). Once exhausted, receive() blocks until the client sends
    end_of_stream, then replays `after_eos`, by default the server's EOS and a normal close.
    """

    def __init__(
        self,
        frames: list[Any],
        *,
        after_eos: list[Any] | None = None,
        close_code: int = 1000,
    ) -> None:
        import aiohttp

        self._ws_type = aiohttp.WSMsgType
        self._frames = list(frames)
        self._after_eos = [EOS, "close"] if after_eos is None else list(after_eos)
        self.close_code = close_code
        self.sent: list[dict[str, Any]] = []
        self._client_eos = asyncio.Event()

    async def send_str(self, data: str) -> None:
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "end_of_stream":
            self._client_eos.set()

    async def receive(self, timeout: float | None = None) -> _Msg:
        if not self._frames:
            await self._client_eos.wait()
            self._frames, self._after_eos = self._after_eos, []
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


def _make_stt(frames: list[Any], *, after_eos: list[Any] | None = None, **kwargs: Any):
    from livekit.plugins.gradium import STT

    ws = FakeWebSocket(frames, after_eos=after_eos)
    session = FakeSession(ws)
    stt = STT(api_key="test-key", http_session=session, **kwargs)  # type: ignore[arg-type]
    return stt, session, ws


def _audio_frame(duration_s: float = 0.08) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=int(SAMPLE_RATE * duration_s),
    )


async def _collect_until(stream, predicate, timeout: float = 3.0) -> list[Any]:
    """Drain events until `predicate` matches or the stream ends."""
    events: list[Any] = []

    async def _drain() -> None:
        async for ev in stream:
            events.append(ev)
            if predicate(ev):
                return

    await asyncio.wait_for(_drain(), timeout)
    return events


def test_stt_default_language():
    """STT defaults to English."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key")

    assert stt._opts.language == "en"


def test_stt_custom_language():
    """STT accepts custom language."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr")

    assert stt._opts.language == "fr"


def test_stream_language_overrides_stream_options():
    """stream language overrides the copied stream config."""
    from livekit.plugins.gradium import STT

    stt = STT(api_key="test-key", language="fr", http_session=MagicMock())

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = stt.stream(language="de")

    assert stream._opts.language == "de"
    assert stt._opts.language == "fr"


@pytest.mark.asyncio
async def test_speech_stream_sends_language_in_json_config():
    """SpeechStream sends language in setup json_config."""
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, LanguageCode
    from livekit.plugins.gradium import STT
    from livekit.plugins.gradium.stt import SpeechStream, STTOptions

    sent_messages: list[str] = []

    class FakeWebSocket:
        async def send_str(self, message: str) -> None:
            sent_messages.append(message)

    class FakeSession:
        async def ws_connect(self, url, headers):
            return FakeWebSocket()

    stt = STT(api_key="test-key")
    opts = STTOptions(language=LanguageCode("pt"), temperature=0.3)

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        task = MagicMock()
        return task

    with patch("livekit.agents.stt.stt.asyncio.create_task", side_effect=_fake_create_task):
        stream = SpeechStream(
            stt=stt,
            opts=opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            api_key="test-key",
            model_endpoint="wss://api.gradium.ai/api/speech/asr",
            model_name="default",
            http_session=FakeSession(),
        )

    await stream._connect_ws()

    setup_msg = json.loads(sent_messages[0])
    assert setup_msg["json_config"] == {"language": "pt", "temp": 0.3}


async def test_final_transcript_uses_delay_in_frames_from_ready():
    """A positive VAD finalizes after the model delay the server announced (3), not 6."""
    frames = [READY, _text("hello", 0.1)] + [_step(i, SILENT) for i in range(4)]
    stt, _, _ = _make_stt(frames)

    stream = stt.stream(conn_options=CONN_OPTS)
    stream.push_frame(_audio_frame())
    try:
        events = await _collect_until(
            stream, lambda ev: ev.type == SpeechEventType.FINAL_TRANSCRIPT
        )
    finally:
        await stream.aclose()

    types = [ev.type for ev in events]
    assert types[0] == SpeechEventType.START_OF_SPEECH
    assert types[-1] == SpeechEventType.FINAL_TRANSCRIPT
    assert events[-1].alternatives[0].text == "hello"


async def test_end_input_sends_eos_and_finalizes_pending_text():
    """Closing the input sends end_of_stream; the server's EOS flushes the last words."""
    stt, _, ws = _make_stt([READY, _text("hello", 0.1), _text("world", 0.4)])

    stream = stt.stream(conn_options=CONN_OPTS)
    stream.push_frame(_audio_frame())
    stream.end_input()
    try:
        events = await _collect_until(stream, lambda ev: False, timeout=5.0)
    finally:
        await stream.aclose()

    assert ws.sent[0]["type"] == "setup"
    assert ws.sent[-1] == {"type": "end_of_stream"}
    finals = [ev for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert [ev.alternatives[0].text for ev in finals] == ["hello world"]
    assert events[-1].type == SpeechEventType.END_OF_SPEECH


async def test_error_frame_raises_api_error():
    error = {"type": "error", "message": "Unsupported language 'xx'.", "code": 1011}
    stt, _, _ = _make_stt([READY, error, "close"])

    stream = stt.stream(conn_options=CONN_OPTS)
    stream.push_frame(_audio_frame())
    try:
        with pytest.raises(APIError, match="Unsupported language"):
            await _collect_until(stream, lambda ev: False)
    finally:
        await stream.aclose()
