from __future__ import annotations

import asyncio
import base64
import gc
import json
import weakref
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
)
from livekit.plugins.sarvam import stt as sarvam_stt

pytestmark = pytest.mark.unit


class _ErrorResponse:
    def __init__(self, *, status: int, error_text: str) -> None:
        self.status = status
        self._error_text = error_text

    async def __aenter__(self) -> _ErrorResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def text(self) -> str:
        return self._error_text


class _ErrorSession:
    def __init__(self, *, status: int, error_text: str) -> None:
        self._status = status
        self._error_text = error_text

    def post(self, *_: object, **__: object) -> _ErrorResponse:
        return _ErrorResponse(status=self._status, error_text=self._error_text)


class _FakeWS:
    def __init__(
        self,
        messages: list[SimpleNamespace],
        *,
        close_code: int | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self._messages = iter(messages)
        self.close_code = close_code
        self._exception = exception
        self.closed = False

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration from None

    async def receive(self) -> SimpleNamespace:
        try:
            return next(self._messages)
        except StopIteration:
            return SimpleNamespace(
                type=sarvam_stt.aiohttp.WSMsgType.CLOSED,
                data=self.close_code,
                extra=None,
            )

    def exception(self) -> BaseException | None:
        return self._exception


def _make_audio_buffer(
    *, num_samples: int = 1600, sample_rate: int = 16000
) -> list[rtc.AudioFrame]:
    data = np.zeros(num_samples, dtype=np.int16).tobytes()
    return [
        rtc.AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
        )
    ]


@pytest.mark.asyncio
async def test_sarvam_stt_recognize_preserves_native_status_error() -> None:
    stt = sarvam_stt.STT(api_key="sk_test")
    stt._session = _ErrorSession(  # type: ignore[assignment]
        status=422, error_text='{"error":"bad request"}'
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stt._recognize_impl(
            _make_audio_buffer(),
            conn_options=APIConnectOptions(
                max_retry=0,
                retry_interval=0.0,
                timeout=1.0,
            ),
        )

    assert excinfo.value.status_code == 422
    assert excinfo.value.body == '{"error":"bad request"}'


@pytest.mark.asyncio
async def test_sarvam_stt_stream_run_preserves_native_api_status_error() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions(
        max_retry=0,
        retry_interval=0.0,
        timeout=1.0,
    )
    stream._connection_lock = asyncio.Lock()
    stream._connection_state = sarvam_stt.ConnectionState.DISCONNECTED
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._client_request_id = None
    stream._server_request_id = None

    async def _raise_native_status_error() -> None:
        raise APIStatusError(
            "native provider error",
            status_code=429,
            body={"error": "rate_limit"},
            retryable=False,
        )

    stream._run_connection = (  # type: ignore[method-assign]
        _raise_native_status_error
    )
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(APIStatusError) as excinfo:
        await stream._run()

    assert excinfo.value.status_code == 429
    assert excinfo.value.body == {"error": "rate_limit"}


@pytest.mark.asyncio
async def test_sarvam_stt_stream_run_does_not_retry_on_timeout() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions(
        max_retry=5,
        retry_interval=0.0,
        timeout=1.0,
    )
    stream._connection_lock = asyncio.Lock()
    stream._connection_state = sarvam_stt.ConnectionState.DISCONNECTED
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._client_request_id = None
    stream._server_request_id = None

    attempts = 0

    async def _raise_timeout() -> None:
        nonlocal attempts
        attempts += 1
        raise asyncio.TimeoutError("simulated timeout")

    stream._run_connection = _raise_timeout  # type: ignore[method-assign]
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(APIConnectionError, match="Failed to connect to STT WebSocket"):
        await stream._run()

    assert attempts == 1


@pytest.mark.asyncio
async def test_sarvam_stt_stream_run_wraps_unexpected_as_status_error() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions(
        max_retry=0,
        retry_interval=0.0,
        timeout=1.0,
    )
    stream._connection_lock = asyncio.Lock()
    stream._connection_state = sarvam_stt.ConnectionState.DISCONNECTED
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._client_request_id = None
    stream._server_request_id = None

    async def _raise_unexpected() -> None:
        raise RuntimeError("unexpected failure")

    stream._run_connection = _raise_unexpected  # type: ignore[method-assign]
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(APIStatusError, match="STT WebSocket session failed: unexpected failure"):
        await stream._run()


@pytest.mark.asyncio
async def test_sarvam_stt_stream_error_message_preserves_raw_payload() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    payload = {
        "type": "error",
        "data": {"message": "invalid model", "code": "400", "details": "model not found"},
    }

    with pytest.raises(APIStatusError) as excinfo:
        await stream._handle_error_message(payload)

    assert excinfo.value.status_code == 400
    assert excinfo.value.body == payload
    assert '"message":"invalid model"' in excinfo.value.message


@pytest.mark.asyncio
async def test_sarvam_stt_process_messages_raises_on_non_json_error_text() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions()
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    ws = _FakeWS(
        [
            SimpleNamespace(
                type=sarvam_stt.aiohttp.WSMsgType.TEXT,
                data="invalid model: saarass:v3",
                extra=None,
            )
        ]
    )

    with pytest.raises(APIStatusError, match="Sarvam STT non-JSON error message"):
        await stream._process_messages(ws)


@pytest.mark.asyncio
async def test_sarvam_stt_process_messages_raises_on_error_close_reason() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions()
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    ws = _FakeWS(
        [
            SimpleNamespace(
                type=sarvam_stt.aiohttp.WSMsgType.CLOSED,
                data=None,
                extra="invalid model: saarass:v3",
            )
        ],
        close_code=1000,
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stream._process_messages(ws)

    assert excinfo.value.body == {
        "msg_type": "CLOSED",
        "close_code": 1000,
        "close_reason": "invalid model: saarass:v3",
    }
    assert "closed unexpectedly" in excinfo.value.message


@pytest.mark.asyncio
async def test_sarvam_stt_process_messages_raises_on_non_graceful_close() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions()
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    ws = _FakeWS(
        [
            SimpleNamespace(
                type=sarvam_stt.aiohttp.WSMsgType.CLOSE,
                data=4000,
                extra="Invalid model 'saarass:v3'.",
            )
        ],
        close_code=None,
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stream._process_messages(ws)

    assert excinfo.value.status_code == 4000
    assert excinfo.value.body == {
        "msg_type": "CLOSE",
        "close_code": 4000,
        "close_reason": "Invalid model 'saarass:v3'.",
    }


@pytest.mark.asyncio
async def test_sarvam_stt_process_messages_raises_on_post_loop_non_graceful_close() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions()
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    ws = _FakeWS([], close_code=4000)

    with pytest.raises(APIStatusError) as excinfo:
        await stream._process_messages(ws)

    assert excinfo.value.status_code == 4000
    assert excinfo.value.body == {
        "msg_type": "CLOSED",
        "close_code": 4000,
        "close_reason": None,
    }


@pytest.mark.asyncio
async def test_sarvam_stt_handle_message_treats_event_error_payload_as_error() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    payload = {
        "type": "event",
        "data": {"event_type": "error", "message": "invalid model", "code": "400"},
    }

    with pytest.raises(APIStatusError) as excinfo:
        await stream._handle_message(payload)

    assert excinfo.value.status_code == 400
    assert excinfo.value.body == payload


@pytest.mark.asyncio
async def test_sarvam_stt_handle_message_treats_unknown_error_shape_as_error() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]
    stream._maybe_set_server_request_id = lambda data: None  # type: ignore[method-assign]

    payload = {
        "type": "status",
        "error": "invalid model",
        "code": "400",
    }

    with pytest.raises(APIStatusError) as excinfo:
        await stream._handle_message(payload)

    assert excinfo.value.status_code == 400
    assert excinfo.value.body == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("liveness_timeout", [3.0, 30.0])
async def test_sarvam_stt_recognize_forces_single_attempt_conn_options(
    monkeypatch: pytest.MonkeyPatch,
    liveness_timeout: float,
) -> None:
    captured = {}

    async def _fake_base_recognize(
        self: object,
        buffer: object,
        *,
        language: object = sarvam_stt.NOT_GIVEN,
        conn_options: APIConnectOptions = (sarvam_stt.DEFAULT_API_CONNECT_OPTIONS),
    ) -> sarvam_stt.stt.SpeechEvent:
        del self, buffer, language
        captured["conn_options"] = conn_options
        return sarvam_stt.stt.SpeechEvent(type=sarvam_stt.stt.SpeechEventType.FINAL_TRANSCRIPT)

    monkeypatch.setattr(sarvam_stt.stt.STT, "recognize", _fake_base_recognize)

    stt = sarvam_stt.STT(api_key="sk_test")
    await stt.recognize(
        _make_audio_buffer(),
        conn_options=APIConnectOptions(
            max_retry=9,
            retry_interval=1.5,
            timeout=7.0,
            liveness_timeout=liveness_timeout,
        ),
    )

    conn_options = captured["conn_options"]
    assert conn_options.max_retry == 0
    assert conn_options.retry_interval == 1.5
    assert conn_options.timeout == 7.0
    assert conn_options.liveness_timeout == liveness_timeout


@pytest.mark.asyncio
@pytest.mark.parametrize("liveness_timeout", [3.0, 30.0])
async def test_sarvam_stt_stream_forces_single_attempt_conn_options(
    monkeypatch: pytest.MonkeyPatch,
    liveness_timeout: float,
) -> None:
    captured = {}

    class _DummyStream:
        def __init__(
            self,
            *,
            stt: object,
            opts: object,
            conn_options: APIConnectOptions,
            api_key: str,
            http_session: object,
            user_keyterms: object,
        ) -> None:
            del stt, opts, api_key, user_keyterms
            captured["conn_options"] = conn_options
            captured["http_session"] = http_session

    monkeypatch.setattr(sarvam_stt, "SpeechStream", _DummyStream)

    stt = sarvam_stt.STT(api_key="sk_test")
    _ = stt.stream(
        conn_options=APIConnectOptions(
            max_retry=8,
            retry_interval=0.2,
            timeout=4.0,
            liveness_timeout=liveness_timeout,
        )
    )

    try:
        conn_options = captured["conn_options"]
        assert conn_options.max_retry == 0
        assert conn_options.retry_interval == 0.2
        assert conn_options.timeout == 4.0
        assert conn_options.liveness_timeout == liveness_timeout
    finally:
        await captured["http_session"].close()


@pytest.mark.asyncio
async def test_sarvam_stt_aclose_closes_tracked_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _idle_run(self: object) -> None:
        del self
        await asyncio.Event().wait()  # cancelled by aclose()

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", _idle_run)

    stt = sarvam_stt.STT(api_key="sk_test")
    stream_a = stt.stream()
    stream_b = stt.stream(language="hi-IN")
    sessions = [stream_a._session, stream_b._session]

    assert all(isinstance(s, sarvam_stt.aiohttp.ClientSession) for s in sessions)
    assert all(not s.closed for s in sessions)
    assert len(stt._streams) == 2

    await stt.aclose()

    assert all(s.closed for s in sessions), (
        "STT.aclose() must close every per-stream aiohttp session"
    )
    assert len(stt._streams) == 0


@pytest.mark.asyncio
async def test_sarvam_stt_async_context_closes_stream_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _idle_run(self: object) -> None:
        del self
        await asyncio.Event().wait()  # cancelled by aclose()

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", _idle_run)

    async with sarvam_stt.STT(api_key="sk_test") as stt:
        stream = stt.stream()
        session = stream._session
        assert not session.closed

    assert session.closed, "exiting `async with` must close per-stream sessions"


@pytest.mark.asyncio
async def test_sarvam_stt_aclose_tolerates_already_closed_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _idle_run(self: object) -> None:
        del self
        await asyncio.Event().wait()  # cancelled by aclose()

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", _idle_run)

    stt = sarvam_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session

    await stream.aclose()
    await stt.aclose()  # must not raise on an already-closed stream

    assert session.closed


@pytest.mark.asyncio
async def test_sarvam_stt_aclose_closes_session_of_garbage_collected_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished stream the caller dropped must not take its session to the GC.

    Regression test for the WeakSet hole in the original fix: with weak
    tracking, a completed ``SpeechStream`` can be collected before
    ``STT.aclose()`` snapshots the set, leaving its per-stream
    ``aiohttp.ClientSession`` unreachable and unclosed. Strong ownership keeps
    the stream (and its session) reachable until ``aclose`` closes it.
    """

    async def _immediate_run(self: object) -> None:
        del self  # return immediately: the stream task finishes on its own

    monkeypatch.setattr(sarvam_stt.SpeechStream, "_run", _immediate_run)

    stt = sarvam_stt.STT(api_key="sk_test")
    stream = stt.stream()
    session = stream._session
    stream_ref = weakref.ref(stream)
    del stream  # drop the caller's reference; only the STT's tracking remains

    # Let the stream task and the base metrics task finish so the stream is at
    # the mercy of the collector (its done-callback cycle needs gc.collect()).
    for _ in range(50):
        if stream_ref() is None:
            break
        await asyncio.sleep(0)
    gc.collect()

    if stream_ref() is None:
        # Weak tracking: the stream really was collected -- the leak scenario.
        assert len(stt._streams) == 0
    else:
        # Strong ownership: the STT retains the finished stream on purpose.
        assert len(stt._streams) == 1

    assert not session.closed, "session must still be open and owned by the STT"

    await stt.aclose()

    assert session.closed, "STT.aclose() must close the session of a stream the caller dropped"


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


async def test_sarvam_stt_stream_update_options_reconnects_instead_of_ending(
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


async def test_sarvam_stt_reconnect_closes_the_open_utterance_on_the_old_connection(
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


async def test_sarvam_stt_reconnect_waits_for_the_flushed_transcript(
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


async def test_sarvam_stt_reconnect_ends_the_turn_when_no_transcript_arrives(
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


async def test_sarvam_stt_reconnect_carries_buffered_audio_over(
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


async def test_sarvam_stt_reconnect_keeps_transcript_times_on_the_stream_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stt = _reconnect_stt(monkeypatch)
    stream = stt.stream()
    await _wait_until(lambda: len(session.sockets) == 1)
    await asyncio.sleep(0.2)

    stream.update_options(language="hi-IN", model="saaras:v4")
    await _wait_until(lambda: len(session.sockets) == 2)
    assert stream.start_time_offset >= 0.2

    await stream._handle_transcript_data(
        {"data": {"transcript": "नमस्ते", "language_code": "hi-IN", "speech_end": 1.7}}
    )
    usage, final = await _next_events(stream, 2)
    assert final.type == sarvam_stt.stt.SpeechEventType.FINAL_TRANSCRIPT
    assert final.alternatives[0].end_time >= 1.9
    await stream.aclose()
    await stt.aclose()
