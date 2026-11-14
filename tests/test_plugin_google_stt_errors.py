from __future__ import annotations

import contextlib
import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core import exceptions as google_exceptions

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
)
from livekit.agents.stt import FallbackAdapter, SpeechEventType
from livekit.plugins.google import stt as google_stt
from tests.fake_stt import FakeSTT

pytestmark = pytest.mark.unit


class _FakeSTT:
    _label = "google.STT"
    model = "latest_long"
    provider = "Google Cloud Platform"

    def emit(self, *_args, **_kwargs):
        pass


class _FakeConnectionPool:
    last_acquire_time = 0.0
    last_connection_reused = False

    def __init__(self, client):
        self._client = client

    @contextlib.asynccontextmanager
    async def connection(self, *, timeout):
        yield self._client

    def remove(self, _client):
        pass


def _default_stt_options() -> google_stt.STTOptions:
    return google_stt.STTOptions(
        languages=[LanguageCode("en-US")],
        detect_language=True,
        interim_results=True,
        punctuate=True,
        spoken_punctuation=False,
        enable_word_time_offsets=True,
        enable_word_confidence=False,
        enable_voice_activity_events=False,
        model="latest_long",
        sample_rate=16000,
        min_confidence_threshold=0.65,
        profanity_filter=False,
    )


def _create_mock_stream(client: object) -> tuple[google_stt.SpeechStream, _FakeConnectionPool]:
    pool = _FakeConnectionPool(client)
    stream = google_stt.SpeechStream(
        stt=_FakeSTT(),  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        pool=pool,
        recognizer_cb=lambda _c: "projects/test/locations/global/recognizers/_",
        config=_default_stt_options(),
    )
    # Cancel auto-started _task so we can control _run execution directly
    stream._task.cancel()
    return stream, pool


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
    """In _recognize_impl, an APIStatusError raised by client.recognize must be
    propagated directly and not caught by except Exception and converted into
    a generic retryable APIConnectionError."""
    mock_client = MagicMock()
    status_err = APIStatusError("Google Speech quota exceeded", status_code=429, retryable=False)
    mock_client.recognize = AsyncMock(side_effect=status_err)

    stt = google_stt.STT(credentials_info={"type": "service_account"}, project="test-proj")
    stt._pool = _FakeConnectionPool(mock_client)

    with pytest.raises(APIStatusError) as exc_info:
        await stt._recognize_impl(_create_audio_buffer(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert exc_info.value is status_err
    assert not isinstance(exc_info.value, APIConnectionError)
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_recognize_impl_preserves_api_error():
    """In _recognize_impl, an APIError raised by client.recognize must be
    propagated directly and not wrapped into APIConnectionError."""
    mock_client = MagicMock()
    api_err = APIError("Custom speech recognition failure", retryable=False)
    mock_client.recognize = AsyncMock(side_effect=api_err)

    stt = google_stt.STT(credentials_info={"type": "service_account"}, project="test-proj")
    stt._pool = _FakeConnectionPool(mock_client)

    with pytest.raises(APIError) as exc_info:
        await stt._recognize_impl(_create_audio_buffer(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert exc_info.value is api_err
    assert not isinstance(exc_info.value, APIConnectionError)
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_speech_stream_preserves_api_status_error():
    """In SpeechStream._run, an APIStatusError raised by streaming_recognize must
    be propagated directly and not converted into generic APIConnectionError."""
    mock_client = MagicMock()
    status_err = APIStatusError("Google STT auth invalid", status_code=401, retryable=False)
    mock_client.streaming_recognize = AsyncMock(side_effect=status_err)

    stream, _ = _create_mock_stream(mock_client)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.status_code == 401
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_preserves_api_error():
    """In SpeechStream._run, an APIError raised by streaming_recognize must
    be propagated directly and not converted into generic APIConnectionError."""
    mock_client = MagicMock()
    api_err = APIError("Unrecoverable stream failure", retryable=False)
    mock_client.streaming_recognize = AsyncMock(side_effect=api_err)

    stream, _ = _create_mock_stream(mock_client)
    try:
        with pytest.raises(APIError) as exc_info:
            await stream._run()

        assert exc_info.value is api_err
        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_process_stream_task_preserves_api_status_error():
    """When the stream reader task in SpeechStream._run raises APIStatusError,
    it must propagate out of _run without being masked as APIConnectionError."""
    status_err = APIStatusError("Google STT permission denied", status_code=403, retryable=False)

    class _ErrorStreamingCall:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise status_err

        def cancel(self):
            pass

    mock_client = MagicMock()
    mock_client.streaming_recognize = AsyncMock(return_value=_ErrorStreamingCall())

    stream, _ = _create_mock_stream(mock_client)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.status_code == 403
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_input_generator_preserves_api_error():
    """When input_generator in SpeechStream encounters an APIStatusError,
    it must NOT catch and swallow it with logger.exception, but propagate it."""
    status_err = APIStatusError("Invalid config in recognizer", status_code=400, retryable=False)

    mock_client = MagicMock()

    class _CaptureCall:
        def __init__(self, requests):
            self._requests = requests

        def __aiter__(self):
            return self

        async def __anext__(self):
            # Drive the input generator
            await self._requests.__anext__()
            raise StopAsyncIteration

        def cancel(self):
            pass

    async def _mock_streaming_recognize(requests):
        return _CaptureCall(requests)

    mock_client.streaming_recognize = AsyncMock(side_effect=_mock_streaming_recognize)

    pool = _FakeConnectionPool(mock_client)

    def _faulty_recognizer_cb(_c):
        raise status_err

    config = dataclasses.replace(_default_stt_options(), model="chirp_2")
    stream = google_stt.SpeechStream(
        stt=_FakeSTT(),  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        pool=pool,
        recognizer_cb=_faulty_recognizer_cb,
        config=config,
    )
    stream._task.cancel()

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_recognize_impl_google_api_call_error_converts_to_api_status_error():
    """In _recognize_impl, GoogleAPICallError (e.g. PermissionDenied code=403) must be
    converted to APIStatusError with status_code=403 and retryable=False, not APIConnectionError."""
    mock_client = MagicMock()
    call_err = google_exceptions.PermissionDenied("Google Cloud IAM permission denied")
    mock_client.recognize = AsyncMock(side_effect=call_err)

    stt = google_stt.STT(credentials_info={"type": "service_account"}, project="test-proj")
    stt._pool = _FakeConnectionPool(mock_client)

    with pytest.raises(APIStatusError) as exc_info:
        await stt._recognize_impl(_create_audio_buffer(), conn_options=DEFAULT_API_CONNECT_OPTIONS)

    assert not isinstance(exc_info.value, APIConnectionError)
    assert exc_info.value.status_code == 403
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_speech_stream_google_api_call_error_converts_to_api_status_error():
    """In SpeechStream._run, GoogleAPICallError (e.g. Unauthenticated code=401) must be
    converted to APIStatusError with status_code=401 and retryable=False, not APIConnectionError."""
    mock_client = MagicMock()
    call_err = google_exceptions.Unauthenticated("Google Cloud authentication failed")
    mock_client.streaming_recognize = AsyncMock(side_effect=call_err)

    stream, _ = _create_mock_stream(mock_client)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._run()

        assert not isinstance(exc_info.value, APIConnectionError)
        assert exc_info.value.status_code == 401
        assert exc_info.value.retryable is False
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_speech_stream_non_retryable_error_does_not_retry():
    """When a non-retryable APIStatusError is encountered, SpeechStream._main_task must
    fail immediately and not retry even when max_retry > 0."""
    mock_client = MagicMock()
    status_err = APIStatusError("Google STT auth invalid", status_code=401, retryable=False)
    mock_client.streaming_recognize = AsyncMock(side_effect=status_err)

    pool = _FakeConnectionPool(mock_client)
    stream = google_stt.SpeechStream(
        stt=_FakeSTT(),  # type: ignore[arg-type]
        conn_options=APIConnectOptions(max_retry=3, timeout=1.0),
        pool=pool,
        recognizer_cb=lambda _c: "projects/test/locations/global/recognizers/_",
        config=_default_stt_options(),
    )

    try:
        with pytest.raises(APIStatusError) as exc_info:
            await stream._task

        assert exc_info.value is status_err
        assert not isinstance(exc_info.value, APIConnectionError)
        # Verify streaming_recognize was only called once, not retried 4 times
        assert mock_client.streaming_recognize.call_count == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_google_stt_fallback_adapter_failover_on_unauthorized():
    """When Google STT encounters a non-retryable APIStatusError (e.g. 401 Unauthorized),
    FallbackAdapter must immediately failover to the backup STT without retrying."""
    status_err = APIStatusError("Google Speech 401 Unauthorized", status_code=401, retryable=False)

    mock_google_client = MagicMock()
    mock_google_client.streaming_recognize = AsyncMock(side_effect=status_err)

    primary_stt = google_stt.STT(
        credentials_info={"type": "service_account"}, project="primary-proj"
    )
    primary_stt._pool = _FakeConnectionPool(mock_google_client)

    backup_stt = FakeSTT(fake_transcript="google fallback succeeded")

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
        and ev.alternatives[0].text == "google fallback succeeded"
        for ev in events
    )
