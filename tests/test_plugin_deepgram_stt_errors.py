from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.stt import FallbackAdapter, SpeechEventType
from livekit.plugins.deepgram import STT
from livekit.plugins.deepgram.stt import SpeechStream
from tests.fake_stt import FakeSTT

pytestmark = pytest.mark.plugin("deepgram")


def _create_mock_stream(session: aiohttp.ClientSession | None = None) -> SpeechStream:
    if session is None:
        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False

    stt = STT(api_key="fake-key", language="en-US", http_session=session)
    stream = SpeechStream(
        stt=stt,
        opts=stt._sanitize_options(),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        api_key="fake-key",
        http_session=session,
        base_url="wss://api.deepgram.com/v1/listen",
    )
    # Cancel the auto-started _task to prevent background execution in tests
    stream._task.cancel()
    if stream._metrics_task is not None:
        stream._metrics_task.cancel()
    return stream


def _create_audio_buffer() -> list[rtc.AudioFrame]:
    frame = rtc.AudioFrame(
        data=b"\x00" * 3200,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=1600,
    )
    return [frame]


@pytest.mark.asyncio
async def test_recognize_impl_preserves_api_status_error():
    """In STT._recognize_impl, an APIStatusError raised during execution
    must be propagated directly and not masked as APIConnectionError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    status_err = APIStatusError("Deepgram unauthorized", status_code=401, retryable=False)

    mock_res = MagicMock()
    mock_res.json = AsyncMock(side_effect=status_err)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_res)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = mock_cm

    stt = STT(api_key="fake-key", http_session=session)

    with pytest.raises(APIStatusError) as exc_info:
        await stt._recognize_impl(_create_audio_buffer(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert exc_info.value is status_err
    assert not isinstance(exc_info.value, APIConnectionError)
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_recognize_impl_preserves_api_error():
    """In STT._recognize_impl, a generic APIError must be propagated directly
    and not wrapped into APIConnectionError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    api_err = APIError("Deepgram quota exhausted", retryable=False)

    mock_res = MagicMock()
    mock_res.json = AsyncMock(side_effect=api_err)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_res)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    session.post.return_value = mock_cm

    stt = STT(api_key="fake-key", http_session=session)

    with pytest.raises(APIError) as exc_info:
        await stt._recognize_impl(_create_audio_buffer(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert exc_info.value is api_err
    assert not isinstance(exc_info.value, APIConnectionError)


@pytest.mark.asyncio
async def test_connect_ws_raises_api_timeout_error_on_timeout():
    """In SpeechStream._connect_ws, an asyncio.TimeoutError must be raised as
    APITimeoutError, NOT masked as APIConnectionError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    async def _timeout_ws_connect(*args, **kwargs):
        raise asyncio.TimeoutError()

    session.ws_connect.side_effect = _timeout_ws_connect

    stream = _create_mock_stream(session)
    try:
        with pytest.raises(APITimeoutError) as exc_info:
            await stream._connect_ws()

        assert type(exc_info.value) is APITimeoutError
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_connect_ws_preserves_api_status_error():
    """In SpeechStream._connect_ws, an APIStatusError raised during connection
    must propagate directly and not be masked as generic APIConnectionError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    status_err = APIStatusError(
        message="Unauthorized Deepgram token", status_code=401, retryable=False
    )
    session.ws_connect.side_effect = status_err

    stream = _create_mock_stream(session)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._connect_ws()

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.status_code == 401
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_connect_ws_preserves_api_error():
    """In SpeechStream._connect_ws, an APIError raised during connection
    must propagate directly and not be wrapped as APIConnectionError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    api_err = APIError("Deepgram connection rejected", retryable=False)
    session.ws_connect.side_effect = api_err

    stream = _create_mock_stream(session)
    try:
        with pytest.raises(APIError) as exc_info:
            await stream._connect_ws()

        assert exc_info.value is api_err
        assert not isinstance(exc_info.value, APIConnectionError)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_main_task_preserves_timeout_error():
    """When _connect_ws times out, SpeechStream._task must raise APITimeoutError."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    async def _timeout_ws_connect(*args, **kwargs):
        raise asyncio.TimeoutError()

    session.ws_connect.side_effect = _timeout_ws_connect

    stt = STT(api_key="fake-key", language="en-US", http_session=session)
    stream = SpeechStream(
        stt=stt,
        opts=stt._sanitize_options(),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        api_key="fake-key",
        http_session=session,
        base_url="wss://api.deepgram.com/v1/listen",
    )

    try:
        with pytest.raises(APITimeoutError) as exc_info:
            await stream._task

        assert type(exc_info.value) is APITimeoutError
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_non_retryable_status_error_no_retries():
    """When SpeechStream encounters a non-retryable APIStatusError (e.g. 401 Unauthorized),
    SpeechStream._main_task must fail immediately without retrying, calling ws_connect only once."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    status_err = APIStatusError("Deepgram 401 Unauthorized", status_code=401, retryable=False)
    session.ws_connect.side_effect = status_err

    stt = STT(api_key="bad-key", language="en-US", http_session=session)
    stream = SpeechStream(
        stt=stt,
        opts=stt._sanitize_options(),
        conn_options=APIConnectOptions(max_retry=3, timeout=1.0),
        api_key="bad-key",
        http_session=session,
        base_url="wss://api.deepgram.com/v1/listen",
    )

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._task

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
        assert session.ws_connect.call_count == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_deepgram_stt_fallback_adapter_failover_on_status_error():
    """When Deepgram STT encounters a non-retryable APIStatusError (e.g. 401 Unauthorized),
    FallbackAdapter must immediately failover to the backup STT without retrying."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False

    status_err = APIStatusError("Deepgram 401 Unauthorized", status_code=401, retryable=False)
    session.ws_connect.side_effect = status_err

    primary_stt = STT(api_key="bad-key", language="en-US", http_session=session)
    backup_stt = FakeSTT(fake_transcript="deepgram fallback succeeded")

    fallback = FallbackAdapter(
        [primary_stt, backup_stt],
        attempt_timeout=2.0,
        max_retry_per_stt=0,
    )
    stream = fallback.stream()

    frame = rtc.AudioFrame(
        data=b"\x00" * 3200,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=1600,
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
        and ev.alternatives[0].text == "deepgram fallback succeeded"
        for ev in events
    )
