"""AirCaps STT plugin against a fake AirCaps server: no network."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, cast

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins import aircaps
from livekit.plugins.aircaps import stt as aircaps_stt

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

_OPTIONS = APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0)
_WS_URL = "wss://rt.example.test/v1/realtime"
_FRAME_BYTES = 640  # 20 ms at 16 kHz


class _FakeServer:
    """One AirCaps session: answers control messages like the real service."""

    def __init__(self, *, start_error: str | None = None) -> None:
        self.start_error = start_error
        self.text = ""
        self.incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.log: list[tuple[str, Any]] = []  # ("text", message) or ("audio", bytes so far)
        self.audio = bytearray()
        self.audio_at: list[tuple[float, int]] = []  # (loop time, bytes so far)
        self.ready = False
        self.closed = False
        self.close_code: int | None = None

    def queue(self, message: dict[str, Any]) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(message), ""))

    def transcript(self, text: str, *, is_final: bool = False, **extra: Any) -> None:
        self.text = text
        self.queue({"type": "transcript", "text": text, "is_final": is_final, **extra})

    def end(self, code: int = 1000) -> None:
        """The server closes the connection."""
        self.close_code = code
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, code, ""))

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [m for k, m in self.log if k == "text" and m["type"] == kind]

    # aiohttp.ClientWebSocketResponse
    async def send_str(self, data: str) -> None:
        message = json.loads(data)
        self.log.append(("text", message))
        kind = message["type"]
        if kind == "start":
            if self.start_error:
                self.queue({"type": "error", "code": self.start_error, "message": "refused"})
                self.end(4400)
            else:
                self.ready = True
                self.queue({"type": "ready"})
        elif kind == "finalize":
            self.transcript(self.text, is_final=True)
        elif kind == "finish":
            self.transcript(self.text, is_final=True)
            self.queue({"type": "finished"})
            self.end()
        elif kind == "ping":
            self.queue({"type": "pong"})

    async def send_bytes(self, data: bytes) -> None:
        assert self.ready, "audio sent before ready"
        self.audio += data
        self.log.append(("audio", len(self.audio)))
        self.audio_at.append((asyncio.get_running_loop().time(), len(self.audio)))

    async def receive(self) -> aiohttp.WSMessage:
        return await self.incoming.get()

    async def ping(self, data: bytes = b"") -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, data, ""))

    async def pong(self, data: bytes = b"") -> None:
        pass

    async def close(self, **kwargs: Any) -> bool:
        self.closed = True
        return True


class _Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeHttp:
    """The AirCaps API: each lookup hands out the next session."""

    def __init__(self, *sessions: _FakeServer, lookup_status: int = 200) -> None:
        self.sessions = deque(sessions)
        self.lookup_status = lookup_status
        self.lookups: list[dict[str, Any]] = []
        self.connects: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.lookups.append({"url": url, **kwargs})
        if self.lookup_status != 200:
            return _Response(
                self.lookup_status, {"error": {"code": "unauthorized", "message": "x"}}
            )
        return _Response(200, {"websocket_url": _WS_URL, "model": "aircaps-live-1"})

    async def ws_connect(self, url: str, **kwargs: Any) -> _FakeServer:
        self.connects.append((url, kwargs))
        return self.sessions.popleft()


def _stt(http: _FakeHttp, **kwargs: Any) -> aircaps.STT:
    return aircaps.STT(api_key="test-key", http_session=cast(aiohttp.ClientSession, http), **kwargs)


def _audio(seconds: float) -> rtc.AudioFrame:
    samples = int(16000 * seconds)
    return rtc.AudioFrame(
        data=b"\x01\x00" * samples, sample_rate=16000, num_channels=1, samples_per_channel=samples
    )


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def _events(stream: aircaps.SpeechStream) -> list[stt.SpeechEvent]:
    return [event async for event in stream]


def _summary(events: list[stt.SpeechEvent]) -> list[tuple[str, str]]:
    return [
        (event.type.value, event.alternatives[0].text if event.alternatives else "")
        for event in events
        if event.type != stt.SpeechEventType.RECOGNITION_USAGE
    ]


START, END = ("start_of_speech", ""), ("end_of_speech", "")


def test_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIRCAPS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AIRCAPS_API_KEY"):
        aircaps.STT()

    monkeypatch.setenv("AIRCAPS_API_KEY", "env-key")
    provider = aircaps.STT()
    assert provider._api_key == "env-key"
    assert provider.model == "aircaps-live-1"
    assert provider.provider == "AirCaps"
    assert provider.capabilities == stt.STTCapabilities(
        streaming=True, interim_results=True, offline_recognize=False, manual_flush=True
    )


async def test_session_setup() -> None:
    server = _FakeServer()
    http = _FakeHttp(server)
    stream = _stt(http).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.1))
    stream.end_input()
    await _events(stream)

    # the address is looked up with the key, and the socket opened with the same key
    assert http.lookups[0]["url"] == "https://api.aircaps.com/v1/realtime/endpoint"
    assert http.lookups[0]["params"] == {"model": "aircaps-live-1"}
    assert http.lookups[0]["headers"] == {"Authorization": "Bearer test-key"}
    url, kwargs = http.connects[0]
    assert url == _WS_URL
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["autoping"] is False
    assert "test-key" not in url

    assert server.log[0] == (
        "text",
        {
            "type": "start",
            "model": "aircaps-live-1",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        },
    )
    assert len(server.audio) == 3200
    assert [m["type"] for k, m in server.log if k == "text"] == ["start", "finalize", "finish"]


async def test_end_of_turn_settings_in_start() -> None:
    server = _FakeServer()
    stream = _stt(
        _FakeHttp(server),
        end_of_turn=True,
        end_of_turn_mode="min_latency",
        max_turn_silence_ms=1000,
    ).stream(conn_options=_OPTIONS)
    stream.end_input()
    await _events(stream)

    assert server.sent("start")[0] == {
        "type": "start",
        "model": "aircaps-live-1",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "end_of_turn": True,
        "end_of_turn_mode": "min_latency",
        "max_turn_silence_ms": 1000,
    }


async def test_turns_from_whole_session_text() -> None:
    server = _FakeServer()
    stream = _stt(_FakeHttp(server)).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.2))
    await _wait_for(lambda: server.audio)

    server.transcript("Hi there")
    server.transcript("Hi there")  # unchanged: no second interim
    stream.flush()
    await _wait_for(lambda: server.sent("finalize"))
    # the next turn's text starts with punctuation that belongs to the previous turn
    server.transcript("Hi there. Book a table", final_chars=9)
    server.transcript("Hi there. Book a table?", is_final=True, end_of_turn=True)
    # punctuation alone is not speech
    server.transcript("Hi there. Book a table?!")
    stream.end_input()
    events = await _events(stream)

    assert _summary(events) == [
        START,
        ("interim_transcript", "Hi there"),
        ("final_transcript", "Hi there"),
        END,
        START,
        ("interim_transcript", "Book a table"),
        ("final_transcript", "Book a table?"),
        END,
        ("final_transcript", ""),  # the answer to the last flush has no words
        ("final_transcript", ""),
    ]
    assert all(e.alternatives[0].language == "en" for e in events if e.alternatives)
    usage = [e for e in events if e.type == stt.SpeechEventType.RECOGNITION_USAGE]
    assert usage and sum(e.recognition_usage.audio_duration for e in usage) == pytest.approx(0.2)


async def test_flush_sends_audio_then_finalize() -> None:
    server = _FakeServer()
    stream = _stt(_FakeHttp(server)).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.03))  # 1.5 frames: the half frame waits for more audio
    stream.flush()
    await _wait_for(lambda: server.sent("finalize"))
    assert server.log[-2:] == [("audio", 960), ("text", {"type": "finalize"})]
    stream.end_input()
    await _events(stream)


@pytest.mark.parametrize("end_of_turn", [False, True])
async def test_only_a_flush_or_the_model_ends_a_turn(end_of_turn: bool) -> None:
    """The plugin never finalizes on its own: interim words wait for a flush or the model."""
    server = _FakeServer()
    stream = _stt(_FakeHttp(server), end_of_turn=end_of_turn).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.2))
    await _wait_for(lambda: server.audio)
    server.transcript("I need a taxi")
    await asyncio.sleep(1.5)  # longer than any wait of the plugin's own
    assert not server.sent("finalize")
    if end_of_turn:
        server.transcript("I need a taxi.", is_final=True, end_of_turn=True, reason="model")
    else:
        stream.flush()
    stream.end_input()
    assert _summary(await _events(stream))[:4] == [
        START,
        ("interim_transcript", "I need a taxi"),
        ("final_transcript", "I need a taxi" + ("." if end_of_turn else "")),
        END,
    ]


async def test_turn_cut_short_by_the_session_ends_and_the_stream_reconnects() -> None:
    first, second = _FakeServer(), _FakeServer()
    http = _FakeHttp(first, second)
    stream = _stt(http).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.2))
    await _wait_for(lambda: first.audio)
    first.transcript("Can you")
    first.end(1011)
    await _wait_for(lambda: second.ready)
    stream.push_frame(_audio(0.2))
    await _wait_for(lambda: second.audio)
    second.transcript("hear me")
    stream.end_input()
    events = await _events(stream)

    assert len(http.lookups) == 2  # the address is looked up again for the new session
    assert _summary(events) == [
        START,
        ("interim_transcript", "Can you"),
        ("final_transcript", "Can you"),
        END,
        START,
        ("interim_transcript", "hear me"),
        ("final_transcript", "hear me"),
        END,
        ("final_transcript", ""),
    ]


async def test_server_ending_the_session_opens_a_new_one() -> None:
    first, second = _FakeServer(), _FakeServer()
    http = _FakeHttp(first, second)
    stream = _stt(http).stream(conn_options=_OPTIONS)
    stream.push_frame(_audio(0.1))
    await _wait_for(lambda: first.audio)
    first.queue({"type": "session_limit", "message": "The 1 h session limit was reached."})
    first.queue({"type": "finished"})
    first.end()
    await _wait_for(lambda: second.ready)
    stream.end_input()
    await _events(stream)
    assert len(http.connects) == 2


async def test_refused_start_is_not_retried() -> None:
    http = _FakeHttp(_FakeServer(start_error="invalid_start"), _FakeServer())
    stream = _stt(http).stream(conn_options=_OPTIONS)
    stream.end_input()
    with pytest.raises(APIStatusError, match="invalid_start") as error:
        await _events(stream)
    assert error.value.retryable is False
    assert len(http.connects) == 1


async def test_refused_key_is_not_retried() -> None:
    http = _FakeHttp(_FakeServer(), lookup_status=401)
    stream = _stt(http).stream(conn_options=_OPTIONS)
    stream.end_input()
    with pytest.raises(APIStatusError) as error:
        await _events(stream)
    assert error.value.status_code == 401
    assert error.value.retryable is False
    assert len(http.lookups) == 1
    assert "test-key" not in str(error.value)


async def test_audio_queued_while_connecting_keeps_its_newest_1_25_s() -> None:
    server = _FakeServer()
    stream = _stt(_FakeHttp(server)).stream(conn_options=_OPTIONS)
    for _ in range(60):  # 3 s queued before the session is ready
        stream.push_frame(_audio(0.05))
    stream.end_input()
    await _events(stream)
    assert len(server.audio) == 40000  # 1.25 s


async def test_audio_is_paced_to_real_time() -> None:
    server = _FakeServer()
    stream = _stt(_FakeHttp(server)).stream(conn_options=_OPTIONS)
    await _wait_for(lambda: server.ready)
    started = asyncio.get_running_loop().time()
    for _ in range(20):  # 2 s pushed at once
        stream.push_frame(_audio(0.1))
    stream.end_input()
    await _events(stream)

    assert len(server.audio) == 64000
    for at, sent in server.audio_at:
        assert sent / 32000 <= at - started + 1.5 + 0.05
    assert server.audio_at[-1][0] - started >= 0.45  # the last 0.5 s went out in real time


async def test_keepalive_ping_while_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aircaps_stt, "_KEEPALIVE_INTERVAL_S", 0.2)
    server = _FakeServer()
    stream = _stt(_FakeHttp(server)).stream(conn_options=_OPTIONS)
    await _wait_for(lambda: server.ready)
    await asyncio.sleep(0.5)
    assert len(server.sent("ping")) == 2
    stream.end_input()
    await _events(stream)


async def test_update_options_configures_open_session() -> None:
    server = _FakeServer()
    provider = _stt(_FakeHttp(server))
    stream = provider.stream(conn_options=_OPTIONS)
    await _wait_for(lambda: server.ready)
    provider.update_options(end_of_turn=True, end_of_turn_threshold=0.7)
    await _wait_for(lambda: server.sent("configure"))
    assert server.sent("configure") == [
        {
            "type": "configure",
            "end_of_turn": True,
            "end_of_turn_mode": "balanced",
            "end_of_turn_threshold": 0.7,
        }
    ]
    assert provider._opts.end_of_turn is True
    stream.end_input()
    await _events(stream)
