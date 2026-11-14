from __future__ import annotations

import asyncio
import base64
import gc
import json
import weakref
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
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
    close_code = 1000

    def __init__(self) -> None:
        self.closed = False
        self.sent: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        message = json.loads(data)
        self.sent.put_nowait(message)
        if message.get("type") == "end_of_stream":
            await self.close()

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
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert (await session.ws.next_message())["type"] == "end_of_stream"
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()
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
        if flush_signal:
            assert await session.ws.next_message() == {"type": "flush"}
        assert (await session.ws.next_message())["type"] == "end_of_stream"
        assert [event async for event in stream] == []
        assert session.ws.sent.empty()


async def test_aclose_does_not_send_end_of_stream(session: _Session) -> None:
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


class _TimestampResponse:
    status = 200

    async def __aenter__(self) -> _TimestampResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self) -> dict[str, object]:
        return {
            "request_id": "r1",
            "transcript": "Sarvam",
            "language_code": "hi-IN",
            "timestamps": {"start_time_seconds": [0.1], "end_time_seconds": [0.4]},
        }


class _RecordingSession:
    def __init__(self) -> None:
        self.fields: dict[str, object] = {}

    def post(self, *, data: object, **_: object) -> _TimestampResponse:
        for type_options, _headers, value in data._fields:  # type: ignore[attr-defined]
            self.fields[type_options["name"]] = value
        return _TimestampResponse()


async def test_recognize_requests_timestamps() -> None:
    stt = sarvam_stt.STT(api_key="sk_test", with_timestamps=True)
    session = _RecordingSession()
    stt._session = session  # type: ignore[assignment]

    event = await stt._recognize_impl(_audio(), conn_options=APIConnectOptions(max_retry=0))

    assert session.fields["with_timestamps"] == "true"
    assert event.alternatives[0].start_time == 0.1
    assert event.alternatives[0].end_time == 0.4


async def test_recognize_omits_timestamps_by_default() -> None:
    stt = sarvam_stt.STT(api_key="sk_test")
    session = _RecordingSession()
    stt._session = session  # type: ignore[assignment]

    await stt._recognize_impl(_audio(), conn_options=APIConnectOptions(max_retry=0))

    assert "with_timestamps" not in session.fields
