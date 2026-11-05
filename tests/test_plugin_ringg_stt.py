"""Hermetic tests for the Ringg STT plugin.

The Ringg WebSocket is faked by monkeypatching the ``ringglabs`` SDK's
``ws_connect``; no network access is needed.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import ringglabs.stt._ws_async as ws_async
from ringglabs.stt import TimeoutConfig

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, Plugin, stt as lk_stt, vad as lk_vad
from livekit.plugins import ringg

pytestmark = pytest.mark.unit

T = lk_stt.SpeechEventType
NO_RETRY = APIConnectOptions(max_retry=0)
FAST = TimeoutConfig(ws_ping_interval=None, ws_recv=0.2)


class _ScriptedWs:
    """Queue-based fake socket: records sends, replays pushed server messages."""

    def __init__(self, first_message: dict[str, object]) -> None:
        self.sent: list[str | bytes] = []
        self.closed = False
        self._queue: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.push(first_message)

    def push(self, payload: dict[str, object] | BaseException) -> None:
        self._queue.put_nowait(json.dumps(payload) if isinstance(payload, dict) else payload)

    async def send(self, payload: str | bytes) -> None:
        self.sent.append(payload)
        if isinstance(payload, str) and json.loads(payload).get("type") == "end":
            from websockets.exceptions import ConnectionClosedOK

            self.push(ConnectionClosedOK(None, None))

    async def recv(self) -> str:
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True

    @property
    def json_sent(self) -> list[dict[str, object]]:
        return [json.loads(p) for p in self.sent if isinstance(p, str)]

    @property
    def audio_bytes(self) -> int:
        return sum(len(p) for p in self.sent if isinstance(p, bytes | bytearray))


def _ready(mode: str = "stream", accept: bool = False) -> dict[str, object]:
    return {
        "type": "ready",
        "request_id": "r1",
        "sample_rate": 16000,
        "language": "hi",
        "mode": mode,
        "enable_cap_punc": True,
        "vad_tail_sil_ms": 200,
        "vad_confidence": 0.55,
        "accept_client_vad_events": accept,
    }


def _transcript(text: str, final: bool) -> dict[str, object]:
    return {"type": "transcript", "transcription": text, "is_final": final, "language": "hi"}


def _install(monkeypatch: pytest.MonkeyPatch, fake: _ScriptedWs) -> None:
    async def fake_connect(uri: str, *args: object, **kwargs: object) -> _ScriptedWs:
        return fake

    monkeypatch.setattr(ws_async, "ws_connect", fake_connect, raising=True)
    monkeypatch.delenv("RINGG_API_KEY", raising=False)


def _frame(ms: int = 20) -> rtc.AudioFrame:
    n = 16000 * ms // 1000
    return rtc.AudioFrame(
        data=bytes(n * 2), sample_rate=16000, num_channels=1, samples_per_channel=n
    )


async def _wait_until(pred: object, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():  # type: ignore[operator]
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _vad_event(type_: lk_vad.VADEventType) -> lk_vad.VADEvent:
    return lk_vad.VADEvent(
        type=type_, samples_index=0, timestamp=0.0, speech_duration=0.0, silence_duration=0.0
    )


class _FakeVADStream(lk_vad.VADStream):
    """START on the first frame, END on flush (end_input flushes)."""

    async def _main_task(self) -> None:
        started = False
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if started:
                    self._event_ch.send_nowait(_vad_event(lk_vad.VADEventType.END_OF_SPEECH))
                    started = False
                continue
            if not started:
                self._event_ch.send_nowait(_vad_event(lk_vad.VADEventType.START_OF_SPEECH))
                started = True


class _FakeVAD(lk_vad.VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=lk_vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> _FakeVADStream:
        return _FakeVADStream(self)


async def _drive(stream: lk_stt.RecognizeStream, fake: _ScriptedWs) -> list[lk_stt.SpeechEvent]:
    """Push one frame, feed interim+final, end input, collect all events."""
    try:
        stream.push_frame(_frame())
        await _wait_until(lambda: fake.audio_bytes > 0)
        fake.push(_transcript("namaste", final=False))
        fake.push(_transcript("namaste duniya", final=True))
        stream.end_input()
        return [ev async for ev in stream]
    finally:
        await stream.aclose()


def test_plugin_is_registered() -> None:
    assert any(p.package == "livekit.plugins.ringg" for p in Plugin.registered_plugins)
    assert ringg.SpeechStream is not None


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RINGG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="RINGG_API_KEY"):
        ringg.STT()


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        ringg.STT(api_key="k", mode="bogus")


async def test_server_endpointing_event_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedWs(_ready())
    _install(monkeypatch, fake)
    stream = ringg.STT(api_key="k", base_url="example.com", timeout=FAST).stream()

    events = await _drive(stream, fake)

    start = fake.json_sent[0]
    assert start["type"] == "start"
    assert start["mode"] == "stream"
    assert start["accept_client_vad_events"] is False
    assert start["sample_rate"] == 16000
    assert start["api_key"] == "k"
    assert fake.audio_bytes == 640
    assert [e.type for e in events] == [
        T.START_OF_SPEECH,
        T.INTERIM_TRANSCRIPT,
        T.FINAL_TRANSCRIPT,
        T.END_OF_SPEECH,
    ]
    assert events[1].alternatives[0].text == "namaste"
    assert events[2].alternatives[0].text == "namaste duniya"
    assert events[2].alternatives[0].language == "hi"
    assert all(e.request_id == "r1" for e in events)
    assert fake.json_sent[-1] == {"type": "end"}
    assert fake.closed is True


async def test_vad_switches_to_client_endpointing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedWs(_ready(mode="on_final", accept=True))
    _install(monkeypatch, fake)
    stream = ringg.STT(api_key="k", base_url="example.com", vad=_FakeVAD(), timeout=FAST).stream()

    events = await _drive(stream, fake)

    assert fake.json_sent[0]["mode"] == "on_final"
    assert fake.json_sent[0]["accept_client_vad_events"] is True
    vad_states = [m["state"] for m in fake.json_sent if m.get("type") == "vad_event"]
    assert vad_states == ["user_start_speaking", "user_stop_speaking"]
    types = [e.type for e in events]
    assert types.count(T.START_OF_SPEECH) == 1
    assert types.count(T.END_OF_SPEECH) == 1
    assert T.FINAL_TRANSCRIPT in types


async def test_language_override_in_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedWs(_ready())
    _install(monkeypatch, fake)
    stream = ringg.STT(api_key="k", base_url="example.com", timeout=FAST).stream(language="en")

    await _drive(stream, fake)

    assert fake.json_sent[0]["language"] == "en"


async def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedWs(_ready())
    _install(monkeypatch, fake)
    monkeypatch.setenv("RINGG_API_KEY", "env-key")
    stream = ringg.STT(timeout=FAST).stream()

    await _drive(stream, fake)

    assert fake.json_sent[0]["api_key"] == "env-key"


async def test_auth_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ScriptedWs(
        {"type": "error", "detail": "bad key", "status_code": 401, "code": "UNAUTHORIZED"}
    )
    _install(monkeypatch, fake)
    stream = ringg.STT(api_key="bad", base_url="example.com", timeout=FAST).stream(
        conn_options=NO_RETRY
    )

    with pytest.raises(APIStatusError) as ei:
        async for _ in stream:
            pass
    await stream.aclose()

    assert ei.value.status_code == 401
    assert ei.value.retryable is False
