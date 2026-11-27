import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIError,
    APIStatusError,
)
from livekit.plugins import assemblyai

pytestmark = pytest.mark.unit


def _create_mock_stream():
    """Helper to create a SpeechStream without starting background connection tasks."""
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_session.closed = False
    stt = assemblyai.STT(api_key="fake-key", http_session=mock_session)

    # We manually construct SpeechStream so _main_task doesn't start automatically
    stream = assemblyai.stt.SpeechStream(
        stt=stt,
        opts=stt._opts,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        api_key="fake-key",
        http_session=mock_session,
        base_url="wss://streaming.assemblyai.com",
    )
    # Cancel the auto-started _main_task and _metrics_task
    stream._task.cancel()
    if stream._metrics_task is not None:
        stream._metrics_task.cancel()
    return stt, stream, mock_session


@pytest.mark.asyncio
async def test_assemblyai_stream_raises_api_status_error_on_handshake_401():
    """When ws_connect fails with HTTP 401/403 ClientResponseError / WSServerHandshakeError,
    _run must raise APIStatusError with status_code=401 and retryable=False,
    not unhandled ClientResponseError."""
    stt, stream, mock_session = _create_mock_stream()

    req_info = MagicMock(spec=aiohttp.RequestInfo)
    mock_session.ws_connect.side_effect = aiohttp.ClientResponseError(
        request_info=req_info,
        history=(),
        status=401,
        message="Unauthorized",
    )

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()
        assert exc_info.value.status_code == 401
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_assemblyai_stream_raises_api_status_error_on_error_message():
    """When AssemblyAI sends an Error frame, SpeechStream must raise APIStatusError
    with the error code and message, and retryable=False for 4xxx client errors,
    rather than ignoring the frame or swallowing it."""
    stt, stream, mock_session = _create_mock_stream()

    error_payload = {
        "type": "Error",
        "error_code": 4001,
        "error": "Authentication failed: invalid API key",
    }
    mock_msg = MagicMock()
    mock_msg.type = aiohttp.WSMsgType.TEXT
    mock_msg.data = json.dumps(error_payload)

    mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    mock_ws.close_code = None
    mock_ws.receive = AsyncMock(return_value=mock_msg)
    mock_ws.send_str = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws._response = MagicMock(status=101)

    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()
        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.status_code == 4001
        assert exc_info.value.retryable is False
        assert "Authentication failed" in exc_info.value.message
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_assemblyai_stream_recv_task_does_not_swallow_api_status_error():
    """When _process_stream_event raises APIStatusError or APIError, recv_task
    must NOT catch and swallow it under `except Exception`, but propagate it directly."""
    stt, stream, mock_session = _create_mock_stream()

    custom_error = APIStatusError(
        message="AssemblyAI quota exceeded", status_code=4002, retryable=False
    )

    mock_msg = MagicMock()
    mock_msg.type = aiohttp.WSMsgType.TEXT
    mock_msg.data = json.dumps({"type": "Turn", "transcript": "hello"})

    mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    mock_ws.close_code = None
    mock_ws.receive = AsyncMock(return_value=mock_msg)
    mock_ws.send_str = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws._response = MagicMock(status=101)

    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    try:
        with patch.object(stream, "_process_stream_event", side_effect=custom_error):
            with pytest.raises(APIStatusError) as exc_info:
                await stream._run()
            assert exc_info.value is custom_error
            assert exc_info.value.status_code == 4002
            assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_assemblyai_stream_preserves_non_retryable_on_websocket_closed_4001():
    """When AssemblyAI closes WebSocket with code 4001 (Unauthorized),
    the raised APIStatusError must have retryable=False so FallbackAdapter fails over
    instead of retrying endlessly."""
    stt, stream, mock_session = _create_mock_stream()

    mock_close_msg = MagicMock()
    mock_close_msg.type = aiohttp.WSMsgType.CLOSED
    mock_close_msg.data = "Unauthorized"
    mock_close_msg.extra = None

    mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    mock_ws.close_code = 4001
    mock_ws.receive = AsyncMock(return_value=mock_close_msg)
    mock_ws.send_str = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws._response = MagicMock(status=101)

    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()
        assert exc_info.value.status_code == 4001
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_assemblyai_stream_preserves_api_error_propagation():
    """Generic APIError raised during SpeechStream._run must not be caught or wrapped
    into APIConnectionError."""
    stt, stream, mock_session = _create_mock_stream()

    api_err = APIError("Custom provider failure", retryable=False)
    mock_session.ws_connect.side_effect = api_err

    try:
        with pytest.raises(APIError) as exc_info:
            await stream._run()
        assert exc_info.value is api_err
        assert not isinstance(exc_info.value, APIConnectionError)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_assemblyai_fallback_adapter_failover_on_unauthorized():
    """FallbackAdapter must catch APIStatusError from AssemblyAI on 401 handshake
    and successfully failover to the backup STT provider."""
    from livekit import rtc
    from livekit.agents.stt import FallbackAdapter, SpeechEventType
    from tests.fake_stt import FakeSTT

    req_info = MagicMock(spec=aiohttp.RequestInfo)
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_session.closed = False
    mock_session.ws_connect.side_effect = aiohttp.ClientResponseError(
        request_info=req_info,
        history=(),
        status=401,
        message="Unauthorized",
    )

    primary_stt = assemblyai.STT(api_key="bad-key", http_session=mock_session)
    backup_stt = FakeSTT(fake_transcript="fallback speech")

    fallback = FallbackAdapter([primary_stt, backup_stt], attempt_timeout=2.0, max_retry_per_stt=0)
    stream = fallback.stream()

    frame = rtc.AudioFrame(
        data=b"\x00" * 3200, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )
    stream.push_frame(frame)
    stream.flush()

    events = []
    async for ev in stream:
        events.append(ev)
        if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
            break

    await stream.aclose()
    await fallback.aclose()

    assert any(
        ev.type == SpeechEventType.FINAL_TRANSCRIPT
        and ev.alternatives
        and ev.alternatives[0].text == "fallback speech"
        for ev in events
    )
