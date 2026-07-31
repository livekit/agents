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
    def __aiter__(self) -> _FakeOutputStream:
        return self

    async def __anext__(self) -> Any:
        await asyncio.Event().wait()
        raise StopAsyncIteration


class _FakeTranscribeStream:
    def __init__(self) -> None:
        self.input_stream = _FakeAudioStream()

    async def await_output(self) -> tuple[None, _FakeOutputStream]:
        return None, _FakeOutputStream()


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
) -> tuple[aws_stt.STT, aws_stt.SpeechStream, _FakeTranscribeStream]:
    transcribe_stream = _FakeTranscribeStream()

    class _FakeTranscribeClient:
        def __init__(self, *, config: Any) -> None:
            self.config = config

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
