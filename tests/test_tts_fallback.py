from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, utils
from livekit.agents.tts import (
    TTS,
    AvailabilityChangedEvent,
    FallbackAdapter,
    SynthesizedAudio,
)
from livekit.agents.types import USERDATA_TTS_STARTED_TIME

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class FallbackAdapterTester(FallbackAdapter):
    def __init__(
        self,
        tts: list[TTS],
        *,
        max_retry_per_tts: int = 1,  # only retry once by default
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            tts,
            max_retry_per_tts=max_retry_per_tts,
            sample_rate=sample_rate,
        )

        self.on("tts_availability_changed", self._on_tts_availability_changed)

        self._availability_changed_ch: dict[int, utils.aio.Chan[AvailabilityChangedEvent]] = {
            id(t): utils.aio.Chan[AvailabilityChangedEvent]() for t in tts
        }

    def _on_tts_availability_changed(self, ev: AvailabilityChangedEvent) -> None:
        self._availability_changed_ch[id(ev.tts)].send_nowait(ev)

    def availability_changed_ch(
        self,
        tts: TTS,
    ) -> utils.aio.ChanReceiver[AvailabilityChangedEvent]:
        return self._availability_changed_ch[id(tts)]


class _NamedTTS(FakeTTS):
    """FakeTTS with a configurable model/provider so tests can tell instances apart."""

    def __init__(self, *, model: str, provider: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._provider_name = provider

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name


def _metrics_listener_count(tts_impl: TTS) -> int:
    return len(tts_impl._events.get("metrics_collected", set()))


def _set_non_streaming(tts_impl: FakeTTS) -> FakeTTS:
    tts_impl._capabilities.streaming = False
    return tts_impl


async def test_stream_closes_temporary_adapter_after_each_request() -> None:
    non_streaming_tts = _set_non_streaming(FakeTTS(fake_audio_duration=1.0))
    fallback_adapter = FallbackAdapterTester([non_streaming_tts, FakeTTS(fake_audio_duration=1.0)])
    baseline = _metrics_listener_count(non_streaming_tts)

    try:
        for _ in range(3):
            async with fallback_adapter.stream() as stream:
                stream.push_text("hello test")
                stream.end_input()
                assert [frame async for frame in stream]

            assert _metrics_listener_count(non_streaming_tts) == baseline
    finally:
        await fallback_adapter.aclose()


async def test_stream_closes_temporary_adapter_after_cancellation() -> None:
    non_streaming_tts = _set_non_streaming(FakeTTS(fake_audio_duration=1.0))
    fallback_adapter = FallbackAdapterTester([non_streaming_tts, FakeTTS(fake_audio_duration=1.0)])
    baseline = _metrics_listener_count(non_streaming_tts)
    stream = fallback_adapter.stream()

    try:
        stream.push_text("hello test")
        stream.flush()
        await asyncio.wait_for(non_streaming_tts.synthesize_ch.recv(), timeout=1.0)
        await stream.aclose()
        assert _metrics_listener_count(non_streaming_tts) == baseline
    finally:
        await stream.aclose()
        await fallback_adapter.aclose()


async def test_stream_reports_active_instance_model_and_provider() -> None:
    fake1 = _NamedTTS(
        model="primary-model",
        provider="primary",
        fake_exception=APIConnectionError("fake1 failed"),
    )
    fake2 = _NamedTTS(model="fallback-model", provider="fallback", fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()

        async for _ in stream:
            pass

    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }

    await fallback_adapter.aclose()


async def test_tts_fallback() -> None:
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=5.0, sample_rate=48000)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.synthesize("hello test") as stream:
        frames = []
        async for data in stream:
            frames.append(data.frame)

        assert fake1.synthesize_ch.recv_nowait()
        assert fake2.synthesize_ch.recv_nowait()

        assert rtc.combine_audio_frames(frames).duration == 5.01

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    fake2.update_options(fake_audio_duration=0.0)

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.synthesize("hello test") as stream:
            async for _ in stream:
                pass

    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    await fallback_adapter.aclose()


async def test_tts_stream_ttfb_includes_failover_time() -> None:
    # the fallback adapter is measured as a single TTS node: ttfb anchors on the first
    # time a sentence was handed to any underlying TTS and is kept when falling back,
    # so time spent failing over counts towards ttfb
    fake1 = FakeTTS(fake_timeout=0.5, fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream(
        conn_options=APIConnectOptions(max_retry=0, retry_interval=0.1)
    ) as stream:
        start = time.perf_counter()
        stream.push_text("hello test")
        stream.end_input()

        frames: list[SynthesizedAudio] = []
        async for data in stream:
            frames.append(data)

    assert frames
    started_times = {f.frame.userdata.get(USERDATA_TTS_STARTED_TIME) for f in frames}
    assert len(started_times) == 1, "all frames should carry the same started time"
    started_time = started_times.pop()
    assert started_time is not None
    # the sentence was handed over at once; fake1 fails without audio after its 0.5s delay,
    # and the anchor stays on that first hand-over rather than moving to fake2's attempt
    assert started_time == pytest.approx(start, abs=0.1)

    await fallback_adapter.aclose()


async def test_tts_chunked_ttfb_includes_failover_time() -> None:
    # same as above for the chunked path: the ChunkedStream base stamps its creation time
    # (when the full text was submitted), which must not be overridden on failover
    fake1 = FakeTTS(fake_timeout=10.0)  # always times out
    fake2 = FakeTTS(fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    start = time.perf_counter()
    async with fallback_adapter.synthesize(
        "hello test",
        conn_options=APIConnectOptions(
            connect_timeout=0.5, read_timeout=0.5, max_retry=0, retry_interval=0.1
        ),
    ) as stream:
        frames: list[SynthesizedAudio] = []
        async for data in stream:
            frames.append(data)

    assert frames
    started_times = {f.frame.userdata.get(USERDATA_TTS_STARTED_TIME) for f in frames}
    assert len(started_times) == 1, "all frames should carry the same started time"
    started_time = started_times.pop()
    assert started_time is not None
    assert started_time == pytest.approx(start, abs=0.1)

    await fallback_adapter.aclose()
