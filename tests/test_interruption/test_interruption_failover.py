"""Tests for interruption detection failover (retry + error-emission) behavior.

Covers:
- WS stream: connection timeout, connection 429, cache-based inference timeout
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents._exceptions import APIError, APIStatusError
from livekit.agents.inference.interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
    InterruptionWebSocketStream,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.types import APIConnectOptions

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

MAX_RETRY = 2
CONN_OPTIONS = APIConnectOptions(max_retry=MAX_RETRY, retry_interval=0.0, timeout=1.0)


def _make_audio_frame(*, num_samples: int = 1600, sample_rate: int = 16000) -> rtc.AudioFrame:
    data = np.zeros(num_samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=num_samples,
    )


def _create_detector(
    mock_session: AsyncMock, *, inference_timeout: float = 0.1
) -> AdaptiveInterruptionDetector:
    return AdaptiveInterruptionDetector(
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=mock_session,
        inference_timeout=inference_timeout,
    )


def _collect_errors(
    detector: AdaptiveInterruptionDetector,
) -> list[InterruptionDetectionError]:
    errors: list[InterruptionDetectionError] = []
    detector.on("error", lambda e: errors.append(e))
    return errors


async def _feed_audio_continuously(
    stream: InterruptionWebSocketStream,
    stop_event: asyncio.Event,
) -> None:
    """Feed overlap audio frames until stop_event is set."""
    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.5, started_at=time.time()))
    while not stop_event.is_set():
        try:
            stream.push_frame(_make_audio_frame())
        except RuntimeError:
            break
        await asyncio.sleep(0.001)


async def _wait_for_stream_failure(
    stream: InterruptionWebSocketStream,
) -> Exception | None:
    """Wait for the stream's background task to complete and return the exception."""
    stop = asyncio.Event()
    feed_task = asyncio.create_task(_feed_audio_continuously(stream, stop))

    try:
        # wait for the internal _task to complete (it will fail after retries)
        try:
            await stream._task
        except Exception as exc:
            return exc
        return None
    finally:
        stop.set()
        await feed_task
        await stream.aclose()


def _mock_request_info() -> MagicMock:
    ri = MagicMock()
    ri.real_url = "http://localhost:9999/bargein"
    ri.method = "GET"
    ri.url = "http://localhost:9999/bargein"
    ri.headers = {}
    return ri


# ---------------------------------------------------------------------------
# WebSocket stream tests
# ---------------------------------------------------------------------------


class TestWsConnectionTimeout:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=asyncio.TimeoutError("connect timeout"))

        detector = _create_detector(mock_session)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1


class TestWsConnection429:
    @pytest.mark.asyncio
    async def test_immediate_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=_mock_request_info(),
                history=(),
                status=429,
                message="Too Many Requests",
            )
        )

        detector = _create_detector(mock_session)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        # 429 -> APIStatusError(retryable=False) so no retries, immediate unrecoverable
        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1


class TestWsCacheTimeout:
    @pytest.mark.asyncio
    async def test_retries_then_emits_unrecoverable(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        inference_timeout = 0.05

        def _make_mock_ws() -> MagicMock:
            mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
            mock_ws.send_str = AsyncMock()
            mock_ws.closed = False
            mock_ws.close_code = None

            async def _slow_send_bytes(*_args: object, **_kwargs: object) -> None:
                await asyncio.sleep(inference_timeout / 2)

            mock_ws.send_bytes = _slow_send_bytes

            async def _receive_hang() -> aiohttp.WSMessage:
                await asyncio.sleep(3600)
                return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

            mock_ws.receive = _receive_hang
            mock_ws.close = AsyncMock(return_value=True)
            return mock_ws

        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: _make_mock_ws())

        detector = _create_detector(mock_session, inference_timeout=inference_timeout)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert exc is not None
        assert isinstance(exc, APIError)

        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1


class TestWsSessionCreatedMissingThreshold:
    @pytest.mark.asyncio
    async def test_immediate_unrecoverable_when_server_omits_threshold(self) -> None:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        def _make_mock_ws() -> MagicMock:
            mock_ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
            mock_ws.send_str = AsyncMock()
            mock_ws.send_bytes = AsyncMock()
            mock_ws.closed = False
            mock_ws.close_code = None

            sent_created = False

            async def _receive() -> aiohttp.WSMessage:
                nonlocal sent_created
                if not sent_created:
                    sent_created = True
                    return aiohttp.WSMessage(
                        type=aiohttp.WSMsgType.TEXT,
                        data=json.dumps({"type": "session.created"}),
                        extra=None,
                    )
                await asyncio.sleep(3600)
                return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

            mock_ws.receive = _receive
            mock_ws.close = AsyncMock(return_value=True)
            return mock_ws

        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: _make_mock_ws())

        detector = _create_detector(mock_session)
        errors = _collect_errors(detector)
        stream = detector.stream(conn_options=CONN_OPTIONS)

        exc = await _wait_for_stream_failure(stream)

        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 500
        assert exc.retryable is False

        # retryable=False -> no retries, immediate unrecoverable
        recoverable_errors = [e for e in errors if e.recoverable]
        unrecoverable_errors = [e for e in errors if not e.recoverable]
        assert len(recoverable_errors) == 0
        assert len(unrecoverable_errors) == 1
