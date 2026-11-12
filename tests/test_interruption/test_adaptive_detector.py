"""AdaptiveInterruptionDetector's websocket stream: session setup, verdicts and failover."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents._exceptions import APIError, APIStatusError
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.inference.interruption import (
    REMOTE_INFERENCE_TIMEOUT,
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
    InterruptionWebSocketStream,
    _AgentSpeechEndedSentinel,
    _AgentSpeechStartedSentinel,
    _OverlapSpeechEndedSentinel,
    _OverlapSpeechStartedSentinel,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import aio

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

CONN_OPTIONS = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=1.0)
RETRY_CONN_OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


def _make_detector(
    ws_connect: AsyncMock | None = None,
    *,
    threshold: NotGivenOr[float] = NOT_GIVEN,
    inference_timeout: float = REMOTE_INFERENCE_TIMEOUT,
) -> AdaptiveInterruptionDetector:
    http_session = AsyncMock(spec=aiohttp.ClientSession)
    http_session.closed = False
    if ws_connect is not None:
        http_session.ws_connect = ws_connect
    return AdaptiveInterruptionDetector(
        threshold=threshold,
        base_url="http://localhost:9999",
        api_key="test-key",
        api_secret="test-secret",
        http_session=http_session,
        inference_timeout=inference_timeout,
    )


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.zeros(1600, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=1600,
    )


def _start_overlap(stream: InterruptionWebSocketStream) -> None:
    stream.push_frame(_AgentSpeechStartedSentinel())
    stream.push_frame(_OverlapSpeechStartedSentinel(speech_duration=0.0, started_at=time.time()))


def _mock_ws(**attrs: object) -> MagicMock:
    ws = MagicMock(spec=aiohttp.ClientWebSocketResponse)
    ws.closed = False
    ws.close_code = None
    ws.send_str = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.close = AsyncMock(return_value=True)
    for name, value in attrs.items():
        setattr(ws, name, value)
    return ws


async def _receive_hang() -> aiohttp.WSMessage:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def test_overlapping_speech_event_does_not_serialize_audio() -> None:
    ev = OverlappingSpeechEvent(type="overlapping_speech")
    ev.speech_input = np.array([1, 2, 3, 4, 5])
    assert ev.model_dump()["speech_input"] is None
    assert ev.model_dump(mode="json")["speech_input"] is None
    assert ev.speech_input is not None


async def test_agent_ended_overlap_is_not_counted_as_backchannel() -> None:
    stream = InterruptionWebSocketStream.__new__(InterruptionWebSocketStream)
    stream._model = MagicMock(model="test-model", provider="test-provider")

    async def _events():
        yield OverlappingSpeechEvent(is_interruption=False, agent_ended=True)

    await stream._metrics_monitor_task(_events())

    metrics = stream._model.emit.call_args.args[1]
    assert metrics.num_interruptions == 0
    assert metrics.num_backchannels == 0


# --- threshold negotiation: sent only when the user overrode it --------------------------


@pytest.mark.parametrize("threshold", [NOT_GIVEN, 0.7])
async def test_session_create_sends_threshold_only_when_overridden(
    threshold: NotGivenOr[float],
) -> None:
    ws = _mock_ws()
    detector = _make_detector(AsyncMock(return_value=ws), threshold=threshold)
    stream = detector.stream(conn_options=CONN_OPTIONS)
    # drive _connect_ws directly, without the live receive loop racing it
    await aio.cancel_and_wait(stream._task)
    try:
        await stream._connect_ws()
    finally:
        await stream.aclose()

    ws.send_str.assert_called_once()
    settings = json.loads(ws.send_str.call_args.args[0])["settings"]
    assert settings.get("threshold") == (None if threshold is NOT_GIVEN else threshold)


@pytest.mark.parametrize(
    ("override", "server_default", "expected"),
    [(0.7, 0.3, 0.7), (NOT_GIVEN, 0.3, 0.3), (NOT_GIVEN, None, None)],
)
async def test_effective_threshold_prefers_override_then_server_default(
    override: NotGivenOr[float], server_default: float | None, expected: float | None
) -> None:
    stream = _make_detector(threshold=override).stream(conn_options=CONN_OPTIONS)
    await aio.cancel_and_wait(stream._task)
    try:
        assert stream._resolve_effective_threshold(server_default) == expected
    finally:
        await stream.aclose()


# --- verdicts belong to the agent speech that requested them ---------------------------


class _FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self.sent_audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.release_audio_send = asyncio.Event()
        self.release_audio_send.set()
        self.session_close_sent = asyncio.Event()
        self._received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        message_type = json.loads(data)["type"]
        if message_type == "session.create":
            self.send_json({"type": "session.created", "default_threshold": 0.5})
        elif message_type == "session.close":
            self.session_close_sent.set()

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio.put_nowait(data)
        await self.release_audio_send.wait()

    async def receive(self) -> aiohttp.WSMessage:
        return await self._received.get()

    async def close(self) -> bool:
        self.closed = True
        return True

    def send_json(self, data: dict[str, object]) -> None:
        self._received.put_nowait(
            aiohttp.WSMessage(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data), extra=None)
        )


async def _send_overlap_audio(stream: InterruptionWebSocketStream, ws: _FakeWebSocket) -> int:
    _start_overlap(stream)
    stream.push_frame(_audio_frame())
    payload = await asyncio.wait_for(ws.sent_audio.get(), timeout=1.0)
    return struct.unpack_from("<Q", payload)[0]


async def test_late_verdict_from_previous_agent_speech_is_ignored() -> None:
    ws = _FakeWebSocket()
    detector = _make_detector(AsyncMock(return_value=ws))
    events: list[OverlappingSpeechEvent] = []
    verdict_received = asyncio.Event()

    def _on_overlap_speech(event: OverlappingSpeechEvent) -> None:
        events.append(event)
        verdict_received.set()

    detector.on("overlapping_speech", _on_overlap_speech)
    stream = detector.stream(conn_options=CONN_OPTIONS)

    async def _cached(request_id: int) -> None:
        while request_id not in stream._cache:
            await asyncio.sleep(0)

    try:
        old_request_id = await _send_overlap_audio(stream, ws)
        await asyncio.wait_for(_cached(old_request_id), timeout=1.0)
        stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time(), agent_ended=True))
        stream.push_frame(_AgentSpeechEndedSentinel())

        new_request_id = await _send_overlap_audio(stream, ws)
        await asyncio.wait_for(_cached(new_request_id), timeout=1.0)
        events.clear()
        verdict_received.clear()

        for request_id, probability in ((old_request_id, 0.99), (new_request_id, 0.75)):
            ws.send_json(
                {
                    "type": "bargein_detected",
                    "created_at": request_id,
                    "probabilities": [probability, probability],
                }
            )
        await asyncio.wait_for(verdict_received.wait(), timeout=1.0)

        [event] = events
        assert event.probability == pytest.approx(0.75)
        assert event.speech_input is not None
    finally:
        await stream.aclose()


async def test_agent_speech_end_invalidates_a_blocked_send() -> None:
    ws = _FakeWebSocket()
    ws.release_audio_send.clear()
    stream = _make_detector(AsyncMock(return_value=ws)).stream(conn_options=CONN_OPTIONS)

    try:
        request_id = await _send_overlap_audio(stream, ws)
        stream.push_frame(_OverlapSpeechEndedSentinel(ended_at=time.time(), agent_ended=True))
        stream.push_frame(_AgentSpeechEndedSentinel())
        stream.end_input()

        async def _agent_speech_ended() -> None:
            while stream._agent_speech_started:
                await asyncio.sleep(0)

        await asyncio.wait_for(_agent_speech_ended(), timeout=1.0)
        ws.release_audio_send.set()
        await asyncio.wait_for(ws.session_close_sent.wait(), timeout=1.0)

        assert request_id not in stream._cache
    finally:
        await stream.aclose()


# --- failover: retries, then exactly one unrecoverable error ----------------------------


def _collect_errors(detector: AdaptiveInterruptionDetector) -> list[InterruptionDetectionError]:
    errors: list[InterruptionDetectionError] = []
    detector.on("error", errors.append)
    return errors


async def _wait_for_stream_failure(stream: InterruptionWebSocketStream) -> BaseException | None:
    """Feed overlap audio until the stream's task fails, and return its exception."""
    stop = asyncio.Event()

    async def _feed() -> None:
        _start_overlap(stream)
        while not stop.is_set():
            try:
                stream.push_frame(_audio_frame())
            except RuntimeError:
                break
            await asyncio.sleep(0.001)

    feed_task = asyncio.create_task(_feed())
    try:
        await stream._task
        return None
    except Exception as exc:
        return exc
    finally:
        stop.set()
        await feed_task
        await stream.aclose()


def _server_ws(*, created: dict[str, object] | None, send_delay: float = 0.0) -> MagicMock:
    """A websocket that optionally answers session.create, then never sends a verdict."""
    sent_created = created is None

    async def _receive() -> aiohttp.WSMessage:
        nonlocal sent_created
        if not sent_created:
            sent_created = True
            return aiohttp.WSMessage(
                type=aiohttp.WSMsgType.TEXT, data=json.dumps(created), extra=None
            )
        return await _receive_hang()

    async def _slow_send_bytes(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(send_delay)

    return _mock_ws(receive=_receive, send_bytes=_slow_send_bytes)


def _rate_limited() -> aiohttp.ClientResponseError:
    request_info = MagicMock(
        real_url="http://localhost:9999/bargein",
        method="GET",
        url="http://localhost:9999/bargein",
        headers={},
    )
    return aiohttp.ClientResponseError(
        request_info=request_info, history=(), status=429, message="Too Many Requests"
    )


@pytest.mark.parametrize(
    "case", ["connect_timeout", "connect_429", "inference_timeout", "missing_threshold"]
)
async def test_stream_failure_emits_one_unrecoverable_error(case: str) -> None:
    inference_timeout = 0.05 if case == "inference_timeout" else 0.1
    if case == "connect_timeout":
        ws_connect = AsyncMock(side_effect=asyncio.TimeoutError("connect timeout"))
    elif case == "connect_429":
        ws_connect = AsyncMock(side_effect=_rate_limited())
    elif case == "inference_timeout":
        # each send outlives half the inference timeout and no verdict ever comes back
        ws_connect = AsyncMock(
            side_effect=lambda *a, **kw: _server_ws(created=None, send_delay=inference_timeout / 2)
        )
    else:
        ws_connect = AsyncMock(
            side_effect=lambda *a, **kw: _server_ws(created={"type": "session.created"})
        )

    detector = _make_detector(ws_connect, inference_timeout=inference_timeout)
    errors = _collect_errors(detector)

    exc = await _wait_for_stream_failure(detector.stream(conn_options=RETRY_CONN_OPTIONS))

    assert isinstance(exc, APIError)
    if case == "missing_threshold":
        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 500
        assert exc.retryable is False
    assert [e.recoverable for e in errors] == [False]


async def test_session_close_reset_is_ignored_during_teardown() -> None:
    session_created = asyncio.Event()
    session_close_attempted = asyncio.Event()
    send_count = 0

    async def _send_str(_data: str) -> None:
        nonlocal send_count
        send_count += 1
        if send_count == 1:
            session_created.set()
            return
        session_close_attempted.set()
        raise ConnectionResetError("Cannot write to closing transport")

    async def _receive() -> aiohttp.WSMessage:
        await session_close_attempted.wait()
        return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSING, data=None, extra=None)

    ws = _mock_ws(send_str=_send_str, receive=_receive)
    detector = _make_detector(AsyncMock(return_value=ws))
    errors = _collect_errors(detector)
    stream = detector.stream(conn_options=RETRY_CONN_OPTIONS)

    try:
        await session_created.wait()
        stream.end_input()
        await asyncio.wait_for(stream._task, timeout=1.0)
    finally:
        await stream.aclose()

    assert send_count == 2
    assert errors == []
    ws.close.assert_awaited_once()


async def test_audio_send_reset_fails_instead_of_hanging() -> None:
    ws = _mock_ws(
        send_bytes=AsyncMock(side_effect=ConnectionResetError("audio transport reset")),
        receive=_receive_hang,
    )
    detector = _make_detector(AsyncMock(return_value=ws))
    errors = _collect_errors(detector)
    stream = detector.stream(conn_options=RETRY_CONN_OPTIONS)
    _start_overlap(stream)
    stream.push_frame(_audio_frame())

    try:
        with pytest.raises(ConnectionResetError, match="audio transport reset"):
            await asyncio.wait_for(asyncio.shield(stream._task), timeout=1.0)
    finally:
        await stream.aclose()

    assert ws.send_bytes.await_count == 1
    assert [e.recoverable for e in errors] == [False]
