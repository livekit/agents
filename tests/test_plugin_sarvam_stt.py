from __future__ import annotations

import asyncio
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
async def test_sarvam_stt_recognize_forces_single_attempt_conn_options(
    monkeypatch: pytest.MonkeyPatch,
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
        ),
    )

    conn_options = captured["conn_options"]
    assert conn_options.max_retry == 0
    assert conn_options.retry_interval == 1.5
    assert conn_options.timeout == 7.0


@pytest.mark.asyncio
async def test_sarvam_stt_stream_forces_single_attempt_conn_options(
    monkeypatch: pytest.MonkeyPatch,
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
        ) -> None:
            del stt, opts, api_key
            captured["conn_options"] = conn_options
            captured["http_session"] = http_session

    monkeypatch.setattr(sarvam_stt, "SpeechStream", _DummyStream)

    stt = sarvam_stt.STT(api_key="sk_test")
    _ = stt.stream(
        conn_options=APIConnectOptions(
            max_retry=8,
            retry_interval=0.2,
            timeout=4.0,
        )
    )

    conn_options = captured["conn_options"]
    assert conn_options.max_retry == 0
    assert conn_options.retry_interval == 0.2
    assert conn_options.timeout == 4.0

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
