from __future__ import annotations

import asyncio
import gc
import weakref

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    http,
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
        messages: list[http.WebSocketMessage],
        *,
        close_code: int | None = None,
    ) -> None:
        self._messages = iter(messages)
        self.close_code = close_code
        self.closed = False

    async def receive(self) -> http.WebSocketMessage:
        return next(self._messages, http.WebSocketMessage(http.WebSocketMessageType.CLOSE, ""))


class _RecordingSession:
    """Stands in for the per-stream ``http.HttpSession`` and records its close."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def recording_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sarvam_stt.http, "HttpSession", _RecordingSession)


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
                read_timeout=1.0,
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
        read_timeout=1.0,
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
        read_timeout=1.0,
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
        raise APITimeoutError("simulated timeout", timeout="connect")

    stream._run_connection = _raise_timeout  # type: ignore[method-assign]
    stream._build_log_context = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(APITimeoutError):
        await stream._run()

    assert attempts == 1


@pytest.mark.asyncio
async def test_sarvam_stt_stream_run_wraps_unexpected_as_status_error() -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._conn_options = APIConnectOptions(
        max_retry=0,
        retry_interval=0.0,
        read_timeout=1.0,
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
        await stream._handle_error_message(sarvam_stt.STT_MESSAGE.validate_python(payload))

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
        [http.WebSocketMessage(http.WebSocketMessageType.TEXT, "invalid model: saarass:v3")]
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
        [http.WebSocketMessage(http.WebSocketMessageType.CLOSE, "invalid model: saarass:v3")],
        close_code=1000,
    )

    with pytest.raises(APIStatusError) as excinfo:
        await stream._process_messages(ws)

    assert excinfo.value.body == {
        "msg_type": "CLOSE",
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
        [http.WebSocketMessage(http.WebSocketMessageType.CLOSE, "Invalid model 'saarass:v3'.")],
        close_code=4000,
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
        "msg_type": "CLOSE",
        "close_code": 4000,
        "close_reason": "",
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
        await stream._handle_message(sarvam_stt.STT_MESSAGE.validate_python(payload))

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
        await stream._handle_message(sarvam_stt.STT_MESSAGE.validate_python(payload))

    assert excinfo.value.status_code == 400
    assert excinfo.value.body == payload


@pytest.mark.asyncio
@pytest.mark.usefixtures("recording_sessions")
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

    assert all(isinstance(s, _RecordingSession) for s in sessions)
    assert all(not s.closed for s in sessions)
    assert len(stt._streams) == 2

    await stt.aclose()

    assert all(s.closed for s in sessions), "STT.aclose() must close every per-stream HTTP session"
    assert len(stt._streams) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("recording_sessions")
async def test_sarvam_stt_aclose_closes_session_of_garbage_collected_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished stream the caller dropped must not take its session to the GC.

    Regression test for the WeakSet hole in the original fix: with weak
    tracking, a completed ``SpeechStream`` can be collected before
    ``STT.aclose()`` snapshots the set, leaving its per-stream
    ``http.HttpSession`` unreachable and unclosed. Strong ownership keeps
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


@pytest.mark.parametrize("as_list", [False, True])
async def test_batch_transcript_without_timestamps_uses_audio_duration(
    as_list: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def recognize(request: web.Request) -> web.Response:
        await request.read()
        return web.json_response({"transcript": "hello", "language_code": "en-IN"})

    app = web.Application()
    app.router.add_post("/recognize", recognize)
    async with TestServer(app) as server, http.open() as session:
        endpoint = str(server.make_url("/recognize"))
        monkeypatch.setattr(sarvam_stt, "_get_urls_for_model", lambda _: (endpoint, endpoint))
        model = sarvam_stt.STT(api_key="test", http_session=session)
        try:
            frame = rtc.AudioFrame.create(
                sample_rate=16000, num_channels=1, samples_per_channel=1600
            )
            result = await model.recognize([frame] if as_list else frame)
            assert result.alternatives[0].end_time == pytest.approx(0.1)
        finally:
            await model.aclose()


@pytest.mark.parametrize("kind", ["data", "events", "event"])
async def test_missing_stream_data_does_not_terminate_recognition(kind: str) -> None:
    stream = object.__new__(sarvam_stt.SpeechStream)
    stream._logger = sarvam_stt.logger.getChild("SpeechStream")
    stream._build_log_context = lambda: {}
    stream._server_request_id = None
    await stream._handle_message(sarvam_stt.STT_MESSAGE.validate_python({"type": kind}))
