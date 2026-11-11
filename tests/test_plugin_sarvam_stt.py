from __future__ import annotations

import asyncio
import base64
import gc
import json
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.plugins import sarvam
from livekit.plugins.sarvam import stt as sarvam_stt

pytestmark = pytest.mark.unit


def _bare_stream(**conn: Any) -> sarvam_stt.SpeechStream:
    """A SpeechStream with what `_run` and the message handlers read, and no running task."""
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions(retry_interval=0.0, timeout=1.0, **conn)
    stream._connection_lock = asyncio.Lock()
    stream._connection_state = sarvam_stt.ConnectionState.DISCONNECTED
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._client_request_id = None
    stream._server_request_id = None
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]
    return stream


def _audio(num_samples: int = 1600) -> list[rtc.AudioFrame]:
    return [rtc.AudioFrame(np.zeros(num_samples, dtype=np.int16).tobytes(), 16000, 1, num_samples)]


async def test_recognize_preserves_native_status_error() -> None:
    class _Response:
        status = 422

        async def __aenter__(self) -> _Response:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def text(self) -> str:
            return '{"error":"bad request"}'

    stt = sarvam_stt.STT(api_key="sk_test")
    stt._session = SimpleNamespace(post=lambda *a, **k: _Response())  # type: ignore[assignment]

    with pytest.raises(APIStatusError) as excinfo:
        await stt._recognize_impl(_audio(), conn_options=APIConnectOptions(max_retry=0))

    assert (excinfo.value.status_code, excinfo.value.body) == (422, '{"error":"bad request"}')


_NATIVE = APIStatusError("native", status_code=429, body={"error": "rate_limit"}, retryable=False)


@pytest.mark.parametrize(
    ("error", "expected", "match"),
    [
        (_NATIVE, APIStatusError, "native"),
        (asyncio.TimeoutError("simulated"), APIConnectionError, "Failed to connect to STT"),
        (RuntimeError("unexpected failure"), APIStatusError, "session failed: unexpected failure"),
    ],
)
async def test_run_fails_once_without_internal_retries(
    error: BaseException, expected: type[Exception], match: str
) -> None:
    stream = _bare_stream(max_retry=5)
    attempts = 0

    async def run_connection() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    stream._run_connection = run_connection  # type: ignore[method-assign]
    with pytest.raises(expected, match=match):
        await stream._run()

    assert attempts == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "error", "data": {"message": "invalid model", "code": "400", "details": "x"}},
        {
            "type": "event",
            "data": {"event_type": "error", "message": "invalid model", "code": "400"},
        },
        {"type": "status", "error": "invalid model", "code": "400"},
    ],
    ids=["error", "event-error", "unknown-shape"],
)
async def test_error_payloads_raise_with_raw_body(payload: dict[str, Any]) -> None:
    with pytest.raises(APIStatusError) as excinfo:
        await _bare_stream()._handle_message(payload)

    assert (excinfo.value.status_code, excinfo.value.body) == (400, payload)


class _FakeWS:
    def __init__(self, messages: list[SimpleNamespace], *, close_code: int | None = None) -> None:
        self._messages = iter(messages)
        self.close_code = close_code
        self.closed = False

    async def receive(self) -> SimpleNamespace:
        return next(
            self._messages,
            SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=self.close_code, extra=None),
        )

    def exception(self) -> BaseException | None:
        return None


async def test_non_json_text_raises() -> None:
    ws = _FakeWS([SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="invalid model", extra=None)])
    with pytest.raises(APIStatusError, match="Sarvam STT non-JSON error message"):
        await _bare_stream()._process_messages(ws)


@pytest.mark.parametrize(
    ("messages", "close_code", "body"),
    [
        # a 1000 close with an error reason is still a failure
        (
            [SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra="invalid model")],
            1000,
            {"msg_type": "CLOSED", "close_code": 1000, "close_reason": "invalid model"},
        ),
        (
            [SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=4000, extra="invalid model")],
            None,
            {"msg_type": "CLOSE", "close_code": 4000, "close_reason": "invalid model"},
        ),
        # the socket iterator ending on a non-graceful close code
        ([], 4000, {"msg_type": "CLOSED", "close_code": 4000, "close_reason": None}),
    ],
    ids=["error-reason", "non-graceful", "after-loop"],
)
async def test_unexpected_close_raises(
    messages: list[SimpleNamespace], close_code: int | None, body: dict[str, Any]
) -> None:
    with pytest.raises(APIStatusError) as excinfo:
        await _bare_stream()._process_messages(_FakeWS(messages, close_code=close_code))

    assert excinfo.value.body == body


async def test_recognize_forces_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, APIConnectOptions] = {}

    async def base_recognize(self: object, buffer: object, **kwargs: Any) -> object:
        captured["conn_options"] = kwargs["conn_options"]
        return sarvam_stt.stt.SpeechEvent(type=sarvam_stt.stt.SpeechEventType.FINAL_TRANSCRIPT)

    monkeypatch.setattr(sarvam_stt.stt.STT, "recognize", base_recognize)
    conn = APIConnectOptions(max_retry=9, retry_interval=1.5, timeout=7.0, liveness_timeout=3.0)
    await sarvam_stt.STT(api_key="sk_test").recognize(_audio(), conn_options=conn)

    assert captured["conn_options"] == APIConnectOptions(
        max_retry=0, retry_interval=1.5, timeout=7.0, liveness_timeout=3.0
    )


async def test_stream_forces_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _DummyStream:
        def __init__(self, *, conn_options: APIConnectOptions, http_session: Any, **_: Any) -> None:
            captured.update(conn_options=conn_options, http_session=http_session)

    monkeypatch.setattr(sarvam_stt, "SpeechStream", _DummyStream)
    conn = APIConnectOptions(max_retry=8, retry_interval=0.2, timeout=4.0, liveness_timeout=3.0)
    sarvam_stt.STT(api_key="sk_test").stream(conn_options=conn)
    await captured["http_session"].close()

    assert captured["conn_options"] == APIConnectOptions(
        max_retry=0, retry_interval=0.2, timeout=4.0, liveness_timeout=3.0
    )


async def test_aclose_closes_every_stream_session(monkeypatch: pytest.MonkeyPatch) -> None:
    async def idle_run(self: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", idle_run)
    stt = sarvam_stt.STT(api_key="sk_test")
    closed_first = stt.stream()
    streams = [closed_first, stt.stream(language="hi-IN")]
    sessions = [s._session for s in streams]
    assert not any(s.closed for s in sessions)

    # aclose tolerates a stream the caller already closed
    await closed_first.aclose()
    await stt.aclose()

    assert all(s.closed for s in sessions)
    assert len(stt._streams) == 0


async def test_aclose_closes_session_of_a_dropped_finished_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_run(self: object) -> None:
        return None

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", immediate_run)
    stt = sarvam_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session
    stream_ref = weakref.ref(stream)
    del stream
    for _ in range(50):
        await asyncio.sleep(0)
    gc.collect()

    # the STT owns finished streams strongly, so a collected stream cannot strand its session
    assert stream_ref() is not None
    assert not session.closed
    await stt.aclose()
    assert session.closed


class _WebSocket:
    """Like Sarvam's endpoint, it never closes the socket on its own."""

    close_code = 1000

    def __init__(self) -> None:
        self.closed = False
        self.sent: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.sent.put_nowait(json.loads(data))

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, 1000, "")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()

    async def next_message(self) -> dict:
        return await asyncio.wait_for(self.sent.get(), timeout=2)


class _Session:
    def __init__(self) -> None:
        self.closed = False
        self.ws = _WebSocket()
        self.connections = 0

    async def ws_connect(self, *args: object, **kwargs: object) -> _WebSocket:
        self.connections += 1
        return self.ws

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _Session:
    monkeypatch.setattr(sarvam_stt, "END_OF_INPUT_GRACE", 0.01)
    session = _Session()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: session)
    return session


def _frame(ms: int, value: int, sample_rate: int) -> rtc.AudioFrame:
    samples = sample_rate * ms // 1000
    return rtc.AudioFrame(bytes([value, 0]) * samples, sample_rate, 1, samples)


async def _expect_audio(ws: _WebSocket, frame: rtc.AudioFrame) -> None:
    received = b""
    while len(received) < len(frame.data.tobytes()):
        message = await ws.next_message()
        assert "type" not in message
        assert message["audio"]["sample_rate"] == frame.sample_rate
        assert message["audio"]["encoding"] == "pcm_s16le"
        chunk = base64.b64decode(message["audio"]["data"])
        assert chunk
        received += chunk
    assert received == frame.data.tobytes()


@pytest.mark.parametrize("flush_signal", [True, False, None])
@pytest.mark.parametrize(("first_ms", "sample_rate"), [(30, 8000), (75, 16000)])
async def test_flush_preserves_audio_and_accepts_next_segment(
    session: _Session, flush_signal: bool | None, first_ms: int, sample_rate: int
) -> None:
    async with sarvam.STT(
        api_key="test-key",
        flush_signal=flush_signal,
        sample_rate=sample_rate,
        input_audio_codec="pcm_s16le",
    ) as model:
        stream = model.stream()
        for value, ms in enumerate([first_ms, 50, 10], start=1):
            frame = _frame(ms, value, sample_rate)
            stream.push_frame(frame)
            stream.flush()
            await _expect_audio(session.ws, frame)
            if flush_signal:
                assert await session.ws.next_message() == {"type": "flush"}
            assert not session.ws.closed

        stream.end_input()
        # one flush closes the input: the framework's (flush_signal) or the stream's own
        assert await session.ws.next_message() == {"type": "flush"}
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()
        assert session.ws.closed
        assert session.connections == 1


@pytest.mark.parametrize("tail_ms", [0, 10])
@pytest.mark.parametrize("flush_signal", [True, False, None])
async def test_end_input_drains_audio_before_terminating(
    session: _Session, tail_ms: int, flush_signal: bool | None
) -> None:
    async with sarvam.STT(
        api_key="test-key", flush_signal=flush_signal, input_audio_codec="pcm_s16le"
    ) as model:
        stream = model.stream()
        tail = _frame(tail_ms, 3, 16000)
        if tail_ms:
            stream.push_frame(tail)
        stream.end_input()
        if tail_ms:
            await _expect_audio(session.ws, tail)
        assert await session.ws.next_message() == {"type": "flush"}
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()
        assert session.ws.closed


async def test_end_input_waits_for_the_open_utterance(session: _Session) -> None:
    """Speech still open at end of input gets its final transcript before the close."""
    async with sarvam.STT(api_key="test-key", input_audio_codec="pcm_s16le") as model:
        stream = model.stream()
        frame = _frame(50, 1, 16000)
        stream.push_frame(frame)
        await _expect_audio(session.ws, frame)
        await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})

        stream.end_input()
        assert await session.ws.next_message() == {"type": "flush"}
        await asyncio.sleep(0.05)
        assert not session.ws.closed  # still waiting on the utterance

        await stream._handle_events({"data": {"signal_type": "END_SPEECH"}})
        await stream._handle_transcript_data(
            {"data": {"transcript": "नमस्ते", "language_code": "hi-IN"}}
        )
        events = [event.type async for event in stream]

        assert events == [
            stt.SpeechEventType.START_OF_SPEECH,
            stt.SpeechEventType.RECOGNITION_USAGE,
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.END_OF_SPEECH,
        ]
        assert session.ws.closed


async def test_end_input_times_out_when_the_utterance_never_finishes(session: _Session) -> None:
    async with sarvam.STT(api_key="test-key", input_audio_codec="pcm_s16le") as model:
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.2))
        frame = _frame(50, 1, 16000)
        stream.push_frame(frame)
        await _expect_audio(session.ws, frame)
        await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})
        stream.end_input()

        with pytest.raises(APITimeoutError, match="final transcript"):
            async for _ in stream:
                pass


async def test_end_input_grace_does_not_count_against_timeout(
    session: _Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout shorter than the grace still lets a clean stream end."""
    monkeypatch.setattr(sarvam_stt, "END_OF_INPUT_GRACE", 0.1)
    async with sarvam.STT(api_key="test-key", input_audio_codec="pcm_s16le") as model:
        stream = model.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.02))
        frame = _frame(50, 1, 16000)
        stream.push_frame(frame)
        await _expect_audio(session.ws, frame)
        stream.end_input()
        assert await session.ws.next_message() == {"type": "flush"}
        assert [event async for event in stream] == []
        assert session.ws.closed


async def test_aclose_does_not_flush(session: _Session) -> None:
    async with sarvam.STT(api_key="test-key", input_audio_codec="pcm_s16le") as model:
        stream = model.stream()
        frame = _frame(50, 1, 16000)
        stream.push_frame(frame)
        await _expect_audio(session.ws, frame)
        await stream.aclose()
        assert session.ws.sent.empty()
        assert session.ws.closed


@pytest.mark.parametrize("flush_signal", [True, False, None])
@pytest.mark.parametrize("client_flushed", [True, False])
async def test_server_end_speech_does_not_flush_next_utterance(
    session: _Session, flush_signal: bool | None, client_flushed: bool
) -> None:
    async with sarvam.STT(
        api_key="test-key", flush_signal=flush_signal, input_audio_codec="pcm_s16le"
    ) as model:
        stream = model.stream()
        first = _frame(50, 1, 16000)
        stream.push_frame(first)
        await _expect_audio(session.ws, first)
        await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})
        if client_flushed:
            stream.flush()
            if flush_signal:
                assert await session.ws.next_message() == {"type": "flush"}
        await stream._handle_events({"data": {"signal_type": "END_SPEECH"}})

        for ms in [50, 10]:
            frame = _frame(ms, 2, 16000)
            stream.push_frame(frame)
            if ms == 10:
                stream.flush()
            await _expect_audio(session.ws, frame)
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert session.ws.sent.empty()
        assert not session.ws.closed


class _ReconnectWS:
    """Like Sarvam's endpoint, it never closes the socket on its own."""

    close_code = 1000

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self._closed = asyncio.Event()
        self.sent: list[dict] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def audio(self) -> bytes:
        return b"".join(base64.b64decode(m["audio"]["data"]) for m in self.sent if "audio" in m)

    async def receive(self) -> sarvam_stt.aiohttp.WSMessage:
        await self._closed.wait()
        return sarvam_stt.aiohttp.WSMessage(sarvam_stt.aiohttp.WSMsgType.CLOSED, 1000, "")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _ReconnectSession:
    def __init__(self) -> None:
        self.closed = False
        self.sockets: list[_ReconnectWS] = []

    async def ws_connect(self, url: str, **_: object) -> _ReconnectWS:
        ws = _ReconnectWS(url)
        self.sockets.append(ws)
        return ws

    async def close(self) -> None:
        self.closed = True


async def test_stream_update_options_reconnects_instead_of_ending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _ReconnectSession()
    monkeypatch.setattr(sarvam_stt.aiohttp, "ClientSession", lambda *a, **k: session)
    stt = sarvam_stt.STT(api_key="sk_test")
    stream = stt.stream()

    async def _wait_for_connections(count: int) -> None:
        while len(session.sockets) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait_for_connections(1), timeout=2)
    stream.update_options(language="hi-IN", model="saaras:v4")

    await asyncio.wait_for(_wait_for_connections(2), timeout=2)
    assert session.sockets[0].closed
    assert "language-code=hi-IN" in session.sockets[1].url
    assert not session.sockets[1].closed

    await stream.aclose()
    await stt.aclose()


def _reconnect_stt(monkeypatch: pytest.MonkeyPatch) -> tuple[_ReconnectSession, sarvam_stt.STT]:
    session = _ReconnectSession()
    monkeypatch.setattr(sarvam_stt.aiohttp, "ClientSession", lambda *a, **k: session)
    return session, sarvam_stt.STT(
        api_key="sk_test", sample_rate=16000, input_audio_codec="pcm_s16le"
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=2)


async def _next_events(stream: sarvam_stt.SpeechStream, count: int) -> list:
    return [await asyncio.wait_for(stream.__anext__(), timeout=2) for _ in range(count)]


def _pcm_frame(ms: int, value: int) -> rtc.AudioFrame:
    samples = 16000 * ms // 1000
    return rtc.AudioFrame(bytes([value, 0]) * samples, 16000, 1, samples)


async def test_reconnect_closes_the_open_utterance_on_the_old_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)

    frame = _pcm_frame(30, 1)  # short of a 50 ms chunk, so it sits in the buffer
    stream.push_frame(frame)
    await _wait_until(lambda: len(stream._audio_buffer) > 0)
    await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})

    stream.update_options(language="hi-IN", model="saaras:v4")
    old = session.sockets[0]
    await _wait_until(lambda: {"type": "flush"} in old.sent)
    assert old.audio() == frame.data.tobytes()
    await stream._handle_transcript_data({"data": {"transcript": "नमस्ते", "language_code": "hi-IN"}})

    await _wait_until(lambda: len(session.sockets) == 2)
    assert old.closed
    events = [event.type for event in await _next_events(stream, 4)]
    assert events == [
        sarvam_stt.stt.SpeechEventType.START_OF_SPEECH,
        sarvam_stt.stt.SpeechEventType.RECOGNITION_USAGE,
        sarvam_stt.stt.SpeechEventType.FINAL_TRANSCRIPT,
        sarvam_stt.stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert not stream._speaking
    await stream.aclose()
    await stt.aclose()


async def test_reconnect_waits_for_the_flushed_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transcript from earlier in the utterance doesn't end the wait for the flush."""
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)
    await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})
    await stream._handle_transcript_data({"data": {"transcript": "पहला", "language_code": "hi-IN"}})

    stream.update_options(language="hi-IN", model="saaras:v4")
    old = session.sockets[0]
    await _wait_until(lambda: {"type": "flush"} in old.sent)
    await asyncio.sleep(0.05)
    assert not old.closed  # still waiting on the flushed audio's transcript
    await stream._handle_transcript_data({"data": {"transcript": "दूसरा", "language_code": "hi-IN"}})

    await _wait_until(lambda: len(session.sockets) == 2)
    events = await _next_events(stream, 6)
    texts = [e.alternatives[0].text for e in events if e.alternatives]
    assert texts == ["पहला", "दूसरा"]
    assert events[-1].type == sarvam_stt.stt.SpeechEventType.END_OF_SPEECH
    await stream.aclose()
    await stt.aclose()


async def test_reconnect_ends_the_turn_when_no_transcript_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sarvam_stt, "RECONNECT_SETTLE_TIMEOUT", 0.05)
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)
    await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})

    stream.update_options(language="hi-IN", model="saaras:v4")
    await _wait_until(lambda: len(session.sockets) == 2)
    events = [event.type for event in await _next_events(stream, 2)]
    assert events == [
        sarvam_stt.stt.SpeechEventType.START_OF_SPEECH,
        sarvam_stt.stt.SpeechEventType.END_OF_SPEECH,
    ]
    await stream.aclose()
    await stt.aclose()


async def test_reconnect_carries_buffered_audio_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)

    first = _pcm_frame(30, 1)
    stream.push_frame(first)
    await _wait_until(lambda: len(stream._audio_buffer) > 0)
    stream.update_options(language="hi-IN", model="saaras:v4")
    await _wait_until(lambda: len(session.sockets) == 2)
    assert session.sockets[0].audio() == b""

    second = _pcm_frame(30, 2)
    stream.push_frame(second)
    new = session.sockets[1]
    await _wait_until(lambda: len(new.audio()) > 0)
    # one 50 ms chunk: all of the first frame, then the start of the second
    assert new.audio() == (first.data.tobytes() + second.data.tobytes())[: 800 * 2]

    await stream.aclose()
    await stt.aclose()


@pytest.mark.virtual_time
async def test_reconnect_keeps_transcript_times_on_the_stream_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)
    stream.push_frame(_pcm_frame(30, 1))
    await _wait_until(lambda: len(stream._audio_buffer) > 0)
    await asyncio.sleep(0.2)

    stream.update_options(language="hi-IN", model="saaras:v4")
    await _wait_until(lambda: len(session.sockets) == 2)
    # the new connection's clock starts with the first audio it reads
    stream.push_frame(_pcm_frame(30, 2))
    await _wait_until(lambda: len(session.sockets[1].audio()) > 0)
    assert stream.start_time_offset >= 0.2

    await stream._handle_transcript_data(
        {"data": {"transcript": "नमस्ते", "language_code": "hi-IN", "speech_end": 1.7}}
    )
    usage, final = await _next_events(stream, 2)
    assert final.type == sarvam_stt.stt.SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].end_time >= 1.9
    await stream.aclose()
    await stt.aclose()
