from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents.metrics import STTMetrics
from livekit.plugins.aws import stt as aws_stt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="AWS Transcribe Streaming SDK requires Python 3.12 or later",
    ),
]


class _FakeAudioStream:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.event_sent = asyncio.Event()
        self.closed = False

    async def send(self, event: Any) -> None:
        self.events.append(event)
        self.event_sent.set()

    async def close(self) -> None:
        self.closed = True


class _FakeOutputStream:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def __aiter__(self) -> _FakeOutputStream:
        return self

    async def __anext__(self) -> Any:
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        await asyncio.Event().wait()
        raise StopAsyncIteration


class _FakeTranscribeStream:
    def __init__(self, output_error: Exception | None = None) -> None:
        self.input_stream = _FakeAudioStream()
        self.configs: list[Any] = []
        self.restarted = asyncio.Event()
        self._output_error = output_error

    async def await_output(self) -> tuple[None, _FakeOutputStream]:
        # the error is one-shot, so only the first attempt fails
        error, self._output_error = self._output_error, None
        return None, _FakeOutputStream(error)


def _frame(duration_ms: int, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def _make_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_error: Exception | None = None,
) -> tuple[aws_stt.STT, aws_stt.SpeechStream, _FakeTranscribeStream]:
    transcribe_stream = _FakeTranscribeStream(output_error)

    class _FakeTranscribeClient:
        def __init__(self, *, config: Any) -> None:
            transcribe_stream.configs.append(config)
            if len(transcribe_stream.configs) > 1:
                transcribe_stream.restarted.set()

        async def start_stream_transcription(self, *, input: Any) -> _FakeTranscribeStream:
            return transcribe_stream

    monkeypatch.setattr(aws_stt, "TranscribeStreamingClient", _FakeTranscribeClient)
    provider = aws_stt.STT(
        region="us-east-1",
        sample_rate=16000,
        credentials=aws_stt.Credentials(
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
        ),
    )
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    return provider, stream, transcribe_stream


def _capture_metrics(provider: aws_stt.STT) -> tuple[list[STTMetrics], asyncio.Event]:
    metrics: list[STTMetrics] = []
    metrics_ready = asyncio.Event()

    def on_metrics_collected(metric: STTMetrics) -> None:
        metrics.append(metric)
        metrics_ready.set()

    provider.on("metrics_collected", on_metrics_collected)
    return metrics, metrics_ready


async def test_aws_stream_emits_periodic_usage_for_sent_audio(monkeypatch: pytest.MonkeyPatch):
    provider, stream, _ = _make_stream(monkeypatch)
    metrics, metrics_ready = _capture_metrics(provider)
    stream._last_audio_duration_report_time = 0.0

    try:
        stream.push_frame(_frame(100))

        event = await asyncio.wait_for(anext(stream), timeout=1.0)

        assert event.type == stt.SpeechEventType.RECOGNITION_USAGE
        assert event.recognition_usage is not None
        assert event.recognition_usage.audio_duration == pytest.approx(0.1)
        await asyncio.wait_for(metrics_ready.wait(), timeout=1.0)
        assert metrics[0].audio_duration == pytest.approx(0.1)
        assert metrics[0].streamed is True
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_aws_stream_flushes_aggregated_usage_without_ending_input(
    monkeypatch: pytest.MonkeyPatch,
):
    provider, stream, _ = _make_stream(monkeypatch)
    metrics, metrics_ready = _capture_metrics(provider)
    stream._last_audio_duration_report_time = time.monotonic()

    try:
        stream.push_frame(_frame(100))
        stream.push_frame(_frame(100))
        stream.flush()

        event = await asyncio.wait_for(anext(stream), timeout=1.0)

        assert event.type == stt.SpeechEventType.RECOGNITION_USAGE
        assert event.recognition_usage is not None
        assert event.recognition_usage.audio_duration == pytest.approx(0.2)
        await asyncio.wait_for(metrics_ready.wait(), timeout=1.0)
        assert len(metrics) == 1
        assert metrics[0].audio_duration == pytest.approx(0.2)
        assert metrics[0].streamed is True
        assert not stream._task.done()
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_aws_stream_flushes_pending_usage_when_input_channel_closes(
    monkeypatch: pytest.MonkeyPatch,
):
    provider, stream, _ = _make_stream(monkeypatch)
    metrics, metrics_ready = _capture_metrics(provider)
    stream._last_audio_duration_report_time = time.monotonic()

    try:
        stream.push_frame(_frame(100))
        stream._input_ch.close()

        event = await asyncio.wait_for(anext(stream), timeout=1.0)

        assert event.type == stt.SpeechEventType.RECOGNITION_USAGE
        assert event.recognition_usage is not None
        assert event.recognition_usage.audio_duration == pytest.approx(0.1)
        await asyncio.wait_for(metrics_ready.wait(), timeout=1.0)
        assert len(metrics) == 1
        assert metrics[0].audio_duration == pytest.approx(0.1)
        assert metrics[0].streamed is True
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_aws_stream_cleanup_survives_closed_event_channel(
    monkeypatch: pytest.MonkeyPatch,
):
    provider, stream, transcribe_stream = _make_stream(monkeypatch)
    stream._last_audio_duration_report_time = time.monotonic()

    try:
        stream.push_frame(_frame(100))
        await asyncio.wait_for(transcribe_stream.input_stream.event_sent.wait(), timeout=1.0)
        stream._event_ch.close()

        await stream.aclose()

        assert transcribe_stream.input_stream.closed is True
    finally:
        if not stream._task.done():
            await stream.aclose()
        await provider.aclose()


async def test_aws_stream_gives_each_attempt_its_own_crt_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    closed: list[Any] = []
    crt_close = aws_stt.AWSCRTHTTPClient.close

    async def tracking_close(self: Any) -> None:
        closed.append(self)
        await crt_close(self)

    monkeypatch.setattr(aws_stt.AWSCRTHTTPClient, "close", tracking_close)

    provider, stream, transcribe_stream = _make_stream(
        monkeypatch,
        output_error=aws_stt.BadRequestException(
            "Your request timed out because no new audio was received for 15 seconds."
        ),
    )

    try:
        stream.push_frame(_frame(100))
        await asyncio.wait_for(transcribe_stream.restarted.wait(), timeout=1.0)

        first, second = transcribe_stream.configs
        # StartStreamTranscription is bidirectional, which only the CRT transport carries
        assert isinstance(first.transport, aws_stt.AWSCRTHTTPClient)
        # AWS finishes the stream on its idle timeout, and the pooled connection dies with
        # it, so the restarted attempt needs a transport of its own
        assert second.transport is not first.transport
        assert second.transport is stream._http_client
        assert closed == [first.transport]
    finally:
        await stream.aclose()
        await provider.aclose()

    assert closed == [first.transport, second.transport]
