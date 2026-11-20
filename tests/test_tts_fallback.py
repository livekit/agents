from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import Agent, APIConnectionError, APIConnectOptions, APIError, utils
from livekit.agents.tts import (
    TTS,
    AudioEmitter,
    AvailabilityChangedEvent,
    ChunkedStream,
    FallbackAdapter,
    StreamAdapter,
    TTSCapabilities,
    TTSError,
)
from livekit.agents.tts.fallback_adapter import FallbackSynthesizeStream, _in_recovery_probe
from livekit.agents.tts.tts import SynthesizedAudio, SynthesizeStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, USERDATA_TTS_STARTED_TIME
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.voice.io import PlaybackFinishedEvent

from .fake_session import FakeActions, create_session, run_session
from .fake_tts import FakeChunkedStream, FakeSynthesizeStream, FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


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


class _NotifyingTTS(FakeTTS):
    """Emits a fallback notice per request, as an inference TTS does on gateway fallback."""

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        self.emit("fallback_activated", text)
        return super().synthesize(text, conn_options=conn_options)


async def test_forwards_fallback_activated_only_for_served_requests() -> None:
    source = _NotifyingTTS(fake_audio_duration=1.0)
    fallback_adapter = FallbackAdapter([source])
    events: list[object] = []
    fallback_adapter.on("fallback_activated", events.append)

    async with fallback_adapter.synthesize("served") as stream:
        assert [frame async for frame in stream]
    # a recovery probe runs with no user attempt in flight
    source.emit("fallback_activated", "probe")
    await fallback_adapter.aclose()
    source.emit("fallback_activated", "after-close")

    assert events == ["served"]


class _ProbeRaceTTS(FakeTTS):
    """Fails its first request, then emits a notice from inside each later request's own task,
    as inference TTS does. The second request is the recovery probe: it emits only once the
    third (the user's retry) is serving the same instance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_serving = asyncio.Event()
        self.probe_emitted = asyncio.Event()
        self._requests = 0

    async def _before_audio(self, request: int) -> None:
        if request == 2:
            await self.user_serving.wait()
            self.emit("fallback_activated", "probe")
            self.probe_emitted.set()
        else:
            self.emit("fallback_activated", "user")
            self.user_serving.set()
            await self.probe_emitted.wait()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        self._requests += 1
        request, race = self._requests, self

        class _Chunked(FakeChunkedStream):
            async def _run(self, output_emitter: AudioEmitter) -> None:
                if request == 1:
                    raise APIConnectionError("primary down")
                await race._before_audio(request)
                await super()._run(output_emitter)

        stream = _Chunked(tts=self, input_text=text, conn_options=conn_options)
        self._synthesize_ch.send_nowait(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        self._requests += 1
        request, race = self._requests, self

        class _Streamed(FakeSynthesizeStream):
            async def _run(self, output_emitter: AudioEmitter) -> None:
                if request == 1:
                    # fail after the input is read, so the adapter has text to probe with
                    async for _ in self._input_ch:
                        pass
                    raise APIConnectionError("primary down")
                await race._before_audio(request)
                await super()._run(output_emitter)

        stream = _Streamed(tts=self, conn_options=conn_options)
        self._stream_ch.send_nowait(stream)
        return stream


@pytest.mark.parametrize("streamed", [False, True])
async def test_probe_notice_not_forwarded_while_user_attempt_serves(streamed: bool) -> None:
    # every instance down: the user's retry serves the instance whose probe is still running
    source = _ProbeRaceTTS(fake_audio_duration=0.5)
    fallback_adapter = FallbackAdapter([source], max_retry_per_tts=0)
    events: list[object] = []
    fallback_adapter.on("fallback_activated", events.append)
    requests = source.stream_ch if streamed else source.synthesize_ch

    async def _request() -> None:
        if streamed:
            async with fallback_adapter.stream() as stream:
                stream.push_text("hello test")
                stream.end_input()
                async for _ in stream:
                    pass
        else:
            async with fallback_adapter.synthesize("hello test") as chunked:
                async for _ in chunked:
                    pass

    with pytest.raises(APIConnectionError):
        await _request()
    assert await asyncio.wait_for(requests.recv(), 1.0)  # the failed attempt
    assert await asyncio.wait_for(requests.recv(), 1.0)  # its recovery probe, parked

    await _request()

    assert source.probe_emitted.is_set()
    assert events == ["user"]
    await fallback_adapter.aclose()


async def test_stream_adapter_forwards_fallback_activated() -> None:
    source = _NotifyingTTS(fake_audio_duration=1.0)
    adapter = StreamAdapter(tts=source)
    events: list[object] = []
    adapter.on("fallback_activated", events.append)

    async with adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()
        assert [frame async for frame in stream]
    await adapter.aclose()
    source.emit("fallback_activated", "after-close")

    assert len(events) == 1
    assert "after-close" not in events


async def test_streamed_fallback_through_temporary_adapter_notifies_once() -> None:
    source = _set_non_streaming(_NotifyingTTS(fake_audio_duration=1.0))
    fallback_adapter = FallbackAdapter([source])
    events: list[object] = []
    fallback_adapter.on("fallback_activated", events.append)

    try:
        async with fallback_adapter.stream() as stream:
            stream.push_text("hello test")
            stream.end_input()
            assert [frame async for frame in stream]
    finally:
        await fallback_adapter.aclose()

    assert len(events) == 1


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


async def test_stream_closes_temporary_adapter_after_failure_and_recovery() -> None:
    non_streaming_tts = _set_non_streaming(
        FakeTTS(fake_exception=APIConnectionError("primary failed"))
    )
    fallback_adapter = FallbackAdapterTester(
        [non_streaming_tts, FakeTTS(fake_audio_duration=1.0)],
        max_retry_per_tts=0,
    )
    baseline = _metrics_listener_count(non_streaming_tts)

    try:
        async with fallback_adapter.stream() as stream:
            stream.push_text("hello test")
            stream.end_input()
            assert [frame async for frame in stream]

        recovery_task = fallback_adapter._status[0].recovering_stream_task
        assert recovery_task is not None
        await recovery_task
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


async def test_reports_active_instance_model_and_provider() -> None:
    fake1 = _NamedTTS(
        model="primary-model",
        provider="primary",
        fake_exception=APIConnectionError("fake1 failed"),
        fake_timeout=0.5,
    )
    fake2 = _NamedTTS(model="fallback-model", provider="fallback", fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    # before any traffic, the primary is reported
    assert fallback_adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }

    async with fallback_adapter.synthesize("hello test") as stream:
        async for _ in stream:
            pass

    # the fallback served the request, so metrics must be labeled with it
    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }
    # once the primary recovers (its recovery task flips it back to available) the next
    # request goes to it first, so that is what model and provider report
    fallback_adapter._status[0].available = True
    assert fallback_adapter.model == "primary-model"
    assert fallback_adapter.provider == "primary"
    fallback_adapter._status[0].available = False
    assert fallback_adapter.model == "fallback-model"

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    # a successful recovery probe must not relabel: the caller never heard its audio
    fake1.update_options(fake_exception=None, fake_audio_duration=1.0)
    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake1).recv(), 1.0)
    ).available, "fake1 should have recovered"

    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }

    # once the recovered primary serves real traffic again, the label follows
    async with fallback_adapter.synthesize("hello again") as stream:
        async for _ in stream:
            pass

    assert fallback_adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }

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


async def test_no_audio() -> None:
    fake1 = FakeTTS(fake_audio_duration=0.0)

    fallback_adapter = FallbackAdapterTester([fake1])

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.synthesize("hello test chunked") as stream:
            async for _ in stream:
                pass

    # stream
    fake1.update_options(fake_audio_duration=5.0)

    async def _input_task(stream: SynthesizeStream, text: str):
        with contextlib.suppress(RuntimeError):
            stream.push_text(text)
            stream.flush()
            stream.end_input()

    async def _stream_task(text: str, audio_duration: float) -> None:
        fake1.update_options(fake_timeout=0.5, fake_audio_duration=audio_duration)

        async with fallback_adapter.stream() as stream:
            input_task = asyncio.create_task(_input_task(stream, text))

            segments = set()
            try:
                async for ev in stream:
                    segments.add(ev.segment_id)
            finally:
                await input_task

            if audio_duration > 0.0:
                assert len(segments) == 1
            else:
                assert len(segments) == 0

    await _stream_task("hello test normal", 1.0)

    with pytest.raises(APIError):
        await _stream_task("hello test no audio", 0.0)

    await fallback_adapter.aclose()


async def test_tts_stream_fallback() -> None:
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()

        async for _ in stream:
            pass

        assert fake1.stream_ch.recv_nowait()
        assert fake2.stream_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    await fallback_adapter.aclose()


async def test_tts_stream_ttfb_includes_failover_time() -> None:
    # the fallback adapter is measured as a single TTS node: ttfb anchors on the first
    # time a sentence was handed to any underlying TTS and is kept when falling back,
    # so time spent failing over counts towards ttfb
    fake1 = FakeTTS(fake_timeout=0.5, fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=5.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream(
        conn_options=APIConnectOptions(timeout=10.0, max_retry=0, retry_interval=0.1)
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
    # fake1 receives the sentence 0.5s in (after its fake connection delay) and fails
    # without audio after two attempts (~1.1s); the anchor must stay on fake1's first
    # attempt rather than moving to fake2's
    assert started_time == pytest.approx(start + 0.5, abs=0.1)

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
        conn_options=APIConnectOptions(timeout=0.5, max_retry=0, retry_interval=0.1),
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


async def test_tts_recover() -> None:
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_exception=APIConnectionError("fake2 failed"), fake_timeout=0.5)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.synthesize("hello test") as stream:
            async for _ in stream:
                pass

        assert fake1.synthesize_ch.recv_nowait()
        assert fake2.synthesize_ch.recv_nowait()

    fake2.update_options(fake_exception=None, fake_audio_duration=5.0)

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake2).recv(), 1.0)
    ).available, "fake2 should have recovered"

    async with fallback_adapter.synthesize("hello test") as stream:
        async for _ in stream:
            pass

    assert fake1.synthesize_ch.recv_nowait()
    assert fake2.synthesize_ch.recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake1).recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake2).recv_nowait()

    await fallback_adapter.aclose()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(("num_channels", "output_rate"), [(1, 48000), (2, 16000), (2, 48000)])
async def test_resampling_preserves_channel_content(
    streaming: bool, num_channels: int, output_rate: int
) -> None:
    input_rate = 16000
    pcm = np.tile(np.array([1000, -2000][:num_channels], dtype=np.int16), input_rate // 5)
    primary = FakeTTS(
        sample_rate=output_rate,
        num_channels=num_channels,
        fake_exception=APIConnectionError("primary unavailable"),
    )
    secondary = FakeTTS(
        sample_rate=input_rate,
        num_channels=num_channels,
        fake_audio_duration=0.2,
        fake_audio_data=pcm.tobytes(),
    )
    adapter = FallbackAdapter([primary, secondary], sample_rate=output_rate, max_retry_per_tts=0)

    async def collect_frames(tts_impl: TTS) -> list[rtc.AudioFrame]:
        if streaming:
            async with tts_impl.stream() as stream:
                stream.push_text("hello test")
                stream.end_input()
                return [event.frame async for event in stream]
        async with tts_impl.synthesize("hello test") as chunked:
            return [event.frame async for event in chunked]

    try:
        expected_frames = await collect_frames(secondary)
        if output_rate != input_rate:
            reference = rtc.AudioResampler(
                input_rate=input_rate, output_rate=output_rate, num_channels=num_channels
            )
            expected_frames = [
                resampled for frame in expected_frames for resampled in reference.push(frame)
            ] + reference.flush()
        actual_frames = await collect_frames(adapter)
        expected = np.frombuffer(
            b"".join(frame.data.tobytes() for frame in expected_frames), dtype=np.int16
        )
        actual = np.frombuffer(
            b"".join(frame.data.tobytes() for frame in actual_frames), dtype=np.int16
        )
        # Independent native resamplers can differ by two int16 quantization steps.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2)
    finally:
        await adapter.aclose()


async def test_audio_resampled() -> None:
    fake1 = FakeTTS(sample_rate=48000, fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=5.0, sample_rate=16000)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.synthesize("hello test") as stream:
        frames: list[SynthesizedAudio] = []
        async for data in stream:
            frames.append(data)

        assert fake1.synthesize_ch.recv_nowait()
        assert fake2.synthesize_ch.recv_nowait()

        assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

        assert frames[-1].is_final is True, "last frame should be final"

        combined_frame = rtc.combine_audio_frames([f.frame for f in frames])
        assert combined_frame.duration == 5.01
        assert combined_frame.sample_rate == 48000

    assert await asyncio.wait_for(fake1.synthesize_ch.recv(), 1.0)

    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()

        frames: list[SynthesizedAudio] = []
        async for data in stream:
            frames.append(data)

        assert fake2.stream_ch.recv_nowait()

        assert frames[-1].is_final is True, "last frame should be final"

        combined_frame = rtc.combine_audio_frames([f.frame for f in frames])
        assert combined_frame.duration == 5.01  # 5.0 + 0.01 final flag
        assert combined_frame.sample_rate == 48000

    await fallback_adapter.aclose()


async def test_timeout():
    fake1 = FakeTTS(fake_timeout=0.5, sample_rate=48000)
    fake2 = FakeTTS(fake_timeout=0.5, sample_rate=48000)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.synthesize(
            "hello test",
            conn_options=APIConnectOptions(timeout=0.1, max_retry=0),
        ) as stream:
            async for _ in stream:
                pass

    assert fake1.synthesize_ch.recv_nowait()
    assert fake2.synthesize_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert await asyncio.wait_for(fake1.synthesize_ch.recv(), 1.0)
    assert await asyncio.wait_for(fake2.synthesize_ch.recv(), 1.0)

    # stream
    with pytest.raises(APIConnectionError):
        async with fallback_adapter.stream(
            conn_options=APIConnectOptions(timeout=0.1, max_retry=0)
        ) as stream:
            stream.push_text("hello test")
            stream.end_input()

            async for _ in stream:
                pass

    assert fake1.stream_ch.recv_nowait()
    assert fake2.stream_ch.recv_nowait()

    assert await asyncio.wait_for(fake1.stream_ch.recv(), 1.0)
    assert await asyncio.wait_for(fake2.stream_ch.recv(), 1.0)

    await fallback_adapter.aclose()


class _GateTTS(FakeTTS):
    """FakeTTS whose recovery probe (the second `stream()`) blocks until `gate` is set.

    Blocking on an event rather than a timer keeps the probe pending for exactly as long as the
    test needs, with nothing racing it to completion.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = asyncio.Event()
        self._stream_calls = 0

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        self._stream_calls += 1
        if self._stream_calls == 1:
            return super().stream(conn_options=conn_options)

        gate = self.gate

        class _Blocking(FakeSynthesizeStream):
            async def _run(self, output_emitter) -> None:  # type: ignore[no-untyped-def]
                await gate.wait()
                await super()._run(output_emitter)

        stream = _Blocking(tts=self, conn_options=conn_options)
        self._stream_ch.send_nowait(stream)
        return stream


async def test_recovery_is_not_suppressed_across_paths() -> None:
    """A recovery probe in flight on one path must not stop the other path from probing.

    The two paths each keep their own recovery slot (as the STT adapter does). With a single
    shared slot, the streamed probe below would sit in it and the chunked request would skip
    recovery entirely, leaving the provider marked unavailable.
    """
    fake1 = _GateTTS(fake_exception=APIConnectionError("fake1 failed"), fake_exception_count=99)
    fake2 = FakeTTS(fake_audio_duration=1.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2], max_retry_per_tts=0)

    # streamed request: fake1 fails only after tokens were pushed, so the streamed recovery
    # actually starts, and then parks on the gate.
    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()

        async for _ in stream:
            pass

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    assert await asyncio.wait_for(fake1.stream_ch.recv(), 1.0)  # the failing attempt
    assert await asyncio.wait_for(fake1.stream_ch.recv(), 1.0)  # the recovery probe, now gated

    # chunked request on the same adapter and provider: it must start its own probe rather than
    # skip recovery because the streamed probe is still running.
    async with fallback_adapter.synthesize("hello test") as chunked:
        async for _ in chunked:
            pass

    assert await asyncio.wait_for(fake1.synthesize_ch.recv(), 1.0), (
        "chunked path did not probe fake1 while the streamed probe was in flight"
    )

    status = fallback_adapter._status[0]
    stream_task = status.recovering_stream_task
    assert stream_task is not None and not stream_task.done()
    assert status.recovering_synthesize_task is not None

    # aclose() must cancel both slots, not just whichever one was written last
    await fallback_adapter.aclose()
    assert stream_task.done()

    fake1.gate.set()


async def test_tts_recover_on_streamed_path() -> None:
    # a skipped provider must still get a recovery probe, or one outage pins the fallback for good
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=1.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async def _drive() -> None:
        async with fallback_adapter.stream() as stream:
            stream.push_text("hello test")
            stream.end_input()
            async for _ in stream:
                pass

    # first request: fake1 fails and is marked unavailable
    await _drive()
    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    fake1.update_options(fake_exception=None, fake_audio_duration=1.0)

    # second request: fake1 is skipped, and must still be probed
    await _drive()

    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake1).recv(), 5.0)
    ).available, "fake1 should have recovered on the streamed path"

    await fallback_adapter.aclose()


async def test_no_recovery_probe_after_close() -> None:
    # aclose() cancels the recovery slots once; a stream still in flight runs
    # its finally afterwards, and a probe started there would have nothing left
    # to cancel it - a live synthesis against a provider that was just closed
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_audio_duration=1.0)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    # first request: fake1 fails and is marked unavailable
    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()
        async for _ in stream:
            pass

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    # second request left mid-flight: _pushed_tokens is populated and fake1 has
    # been skipped, so its probe is pending when the adapter closes
    fake1.update_options(fake_timeout=30.0, fake_exception=None)
    stream = fallback_adapter.stream()
    stream.push_text("hello test")  # deliberately no end_input()

    drain = asyncio.create_task(_drain_stream(stream))
    await asyncio.sleep(0.5)

    await fallback_adapter.aclose()
    await utils.aio.cancel_and_wait(drain)
    await stream.aclose()

    for tts_status in fallback_adapter._status:
        for task in (tts_status.recovering_synthesize_task, tts_status.recovering_stream_task):
            assert task is None or task.done(), "a recovery probe outlived aclose()"


async def _drain_stream(stream: SynthesizeStream) -> None:
    with contextlib.suppress(Exception):
        async for _ in stream:
            pass


_SECONDS_PER_CHAR = 0.05
_REPLY_TOKENS = ["Your ", "order ", "has ", "shipped. ", "It ", "arrives ", "Monday."]
_REPLY = "".join(_REPLY_TOKENS)


class _OutageTTS(TTS):
    """Streaming TTS that refuses connections for `outage` seconds from its first connection
    attempt, failing each stream before it reads any text. Once it is back, it records the text
    each stream was given and speaks it at 50ms per character."""

    def __init__(self, *, outage: float) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )
        self._outage = outage
        self._down_until: float | None = None
        self.received: list[str] = []

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return _OutageStream(tts=self, conn_options=conn_options)


class _OutageStream(SynthesizeStream):
    async def _run(self, output_emitter: AudioEmitter) -> None:
        assert isinstance(self._tts, _OutageTTS)
        await asyncio.sleep(0.05)  # connecting
        now = asyncio.get_running_loop().time()
        if self._tts._down_until is None:
            self._tts._down_until = now + self._tts._outage
        if now < self._tts._down_until:
            raise APIConnectionError("connection refused")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        text = ""
        async for data in self._input_ch:
            if isinstance(data, str):
                text += data
                output_emitter.push(b"\x00\x00" * int(24000 * _SECONDS_PER_CHAR * len(data)))
        output_emitter.end_segment()
        self._tts.received.append(text)


async def _speak_reply_on(stream: SynthesizeStream) -> None:
    # like Agent.default.tts_node: the LLM tokens still arrive while the TTS is down
    async def _push_reply() -> None:
        for token in _REPLY_TOKENS:
            await asyncio.sleep(0.2)
            stream.push_text(token)
        stream.end_input()

    push_task = asyncio.create_task(_push_reply())
    try:
        async with stream:
            async for _ in stream:
                pass
    finally:
        await utils.aio.cancel_and_wait(push_task)


async def _speak_reply(fallback_adapter: FallbackAdapter) -> None:
    # AgentSession's default tts_conn_options (max_retry=3)
    await _speak_reply_on(fallback_adapter.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS))


async def test_outage_longer_than_the_instance_retries_is_not_retried_by_the_adapter() -> None:
    # the primary is down for the whole call and the secondary for 3s from its first connection
    # attempt, longer than its own retries (max_retry_per_tts=2, ~2.2s). The adapter does not
    # retry the request itself, so it fails once, with one unrecoverable error. Before, the
    # session's max_retry=3 made it rerun both instances' retries and the secondary then served
    primary = _OutageTTS(outage=float("inf"))
    secondary = _OutageTTS(outage=3.0)
    fallback_adapter = FallbackAdapter([primary, secondary])
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(APIConnectionError):
        await _speak_reply(fallback_adapter)
    elapsed = loop.time() - start
    await fallback_adapter.aclose()

    assert [e.recoverable for e in errors] == [False]
    assert primary.received == [] and secondary.received == []
    assert elapsed < 6.0, "the adapter reran the instances' retries"


class _ReadThenFailTTS(TTS):
    """Streaming TTS whose first stream reads two tokens and then fails, before any audio. Later
    streams record the text they are given and speak it. Recovery probes are not counted."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )
        self.streams = 0
        self.received: list[str] = []

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        probe = _in_recovery_probe.get()
        if not probe:
            self.streams += 1
        stream = _ReadThenFailStream(tts=self, conn_options=conn_options)
        stream.probe = probe
        return stream


class _ReadThenFailStream(SynthesizeStream):
    probe = False

    async def _run(self, output_emitter: AudioEmitter) -> None:
        assert isinstance(self._tts, _ReadThenFailTTS)
        first = not self.probe and self._tts.streams == 1
        text = ""
        tokens = 0
        if first:
            async for data in self._input_ch:
                if isinstance(data, str):
                    tokens += 1
                    if tokens == 2:
                        raise APIConnectionError("connection dropped")

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        async for data in self._input_ch:
            if isinstance(data, str):
                text += data
                output_emitter.push(b"\x00\x00" * int(24000 * _SECONDS_PER_CHAR * len(data)))
        output_emitter.end_segment()
        if not self.probe:
            self._tts.received.append(text)


async def test_retried_stream_sends_text_once() -> None:
    # FallbackAdapter.stream() never retries, but FallbackSynthesizeStream is exported and
    # SynthesizeStream retries _run when it is built with max_retry > 0. The retry replays the
    # whole input, so the text the first attempt recorded must not be sent again
    tts_impl = _ReadThenFailTTS()
    fallback_adapter = FallbackAdapter([tts_impl], max_retry_per_tts=0)
    options = APIConnectOptions(max_retry=1, timeout=10.0, retry_interval=1.5)

    await _speak_reply_on(FallbackSynthesizeStream(tts=fallback_adapter, conn_options=options))
    await fallback_adapter.aclose()

    assert tts_impl.streams == 2  # the first attempt, then the retry
    assert tts_impl.received == [_REPLY]


async def test_session_speaks_reply_once_after_tts_outage() -> None:
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Where is my order?")
    actions.add_llm(_REPLY)

    # the outage starts with the reply and ends within the adapter's retries of the TTS, as
    # above; the session keeps its default conn_options
    fallback_adapter = FallbackAdapter([_OutageTTS(outage=1.0)])
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    session = create_session(actions)
    playback_finished_events: list[PlaybackFinishedEvent] = []
    session.output.audio.on("playback_finished", playback_finished_events.append)

    agent = Agent(instructions="You are a helpful assistant.", tts=fallback_adapter)
    await asyncio.wait_for(run_session(session, agent), timeout=60)
    await fallback_adapter.aclose()

    assert errors == []
    played = [ev.playback_position for ev in playback_finished_events]
    assert played == pytest.approx([len(_REPLY) * _SECONDS_PER_CHAR], abs=0.1)


class _CountingTTS(FakeTTS):
    """FakeTTS that keeps the requests made for users, leaving out recovery probes. Each
    request also appends the instance's `name` to `calls`, if given, to show the order."""

    def __init__(self, *, name: str = "", calls: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_requests: list[FakeChunkedStream | FakeSynthesizeStream] = []
        self._name = name
        self._calls = calls

    @property
    def user_attempts(self) -> int:
        return sum(request.attempt for request in self.user_requests)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        stream = super().synthesize(text, conn_options=conn_options)
        self._record(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        stream = super().stream(conn_options=conn_options)
        self._record(stream)
        return stream

    def _record(self, request: FakeChunkedStream | FakeSynthesizeStream) -> None:
        if not _in_recovery_probe.get():
            self.user_requests.append(request)
            if self._calls is not None:
                self._calls.append(self._name)


async def _request(
    fallback_adapter: FallbackAdapter, *, streamed: bool, text: str = "hello test", **kwargs: Any
) -> None:
    if streamed:
        async with fallback_adapter.stream(**kwargs) as stream:
            if text:
                stream.push_text(text)
            stream.end_input()
            async for _ in stream:
                pass
    else:
        async with fallback_adapter.synthesize(text, **kwargs) as chunked:
            async for _ in chunked:
                pass


@pytest.mark.parametrize("streamed", [False, True])
async def test_caller_max_retry_is_ignored(streamed: bool) -> None:
    # AgentSession passes max_retry=3 on every request. Applied to the adapter, it reruns every
    # instance's retries once more per attempt (4 x 3 = 12 attempts per instance here)
    fake1 = _CountingTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = _CountingTTS(fake_exception=APIConnectionError("fake2 failed"))
    fallback_adapter = FallbackAdapter([fake1, fake2], max_retry_per_tts=2)
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    caller_options = APIConnectOptions(max_retry=3, timeout=0.5, retry_interval=0.1)
    with pytest.raises(APIConnectionError):
        await _request(fallback_adapter, streamed=streamed, conn_options=caller_options)

    for fake in (fake1, fake2):
        assert fake.user_attempts == 3  # the instance's own retries, once
        for request in fake.user_requests:
            assert request._conn_options.max_retry == 2
            # the caller's timeout and retry interval still reach the instances
            assert request._conn_options.timeout == 0.5
            assert request._conn_options.retry_interval == 0.1

    await fallback_adapter.aclose()


@pytest.mark.parametrize("streamed", [False, True])
async def test_all_instances_failing_emits_one_unrecoverable_error(streamed: bool) -> None:
    fake1 = _CountingTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = _CountingTTS(fake_exception=APIConnectionError("fake2 failed"))
    fallback_adapter = FallbackAdapter([fake1, fake2], max_retry_per_tts=1)
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    # the options AgentSession passes: max_retry=3
    with pytest.raises(APIConnectionError):
        await _request(
            fallback_adapter, streamed=streamed, conn_options=DEFAULT_API_CONNECT_OPTIONS
        )

    assert [e.recoverable for e in errors] == [False]
    assert isinstance(errors[0].error, APIConnectionError)
    assert [fake1.user_attempts, fake2.user_attempts] == [2, 2]  # each tried once: 1 + 1 retry

    await fallback_adapter.aclose()


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("unavailable", [[], [1], [0, 1]])
async def test_every_instance_is_tried_once_before_a_request_fails(
    streamed: bool, unavailable: list[int]
) -> None:
    # instances marked unavailable are tried last, not skipped: a request does not fail while
    # one is left untried
    calls: list[str] = []
    fake1 = _CountingTTS(name="fake1", calls=calls, fake_exception=APIConnectionError("1 failed"))
    fake2 = _CountingTTS(name="fake2", calls=calls, fake_exception=APIConnectionError("2 failed"))
    fallback_adapter = FallbackAdapter([fake1, fake2], max_retry_per_tts=0)
    for index in unavailable:
        fallback_adapter._status[index].available = False
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    with pytest.raises(APIConnectionError):
        await _request(fallback_adapter, streamed=streamed)

    assert calls == ["fake1", "fake2"]
    assert [e.recoverable for e in errors] == [False]
    assert [fake1.user_attempts, fake2.user_attempts] == [1, 1]

    await fallback_adapter.aclose()


@pytest.mark.parametrize("streamed", [False, True])
async def test_failure_after_audio_emits_recoverable_error_and_probes(streamed: bool) -> None:
    # fake1 fails once its audio has started; the utterance ends there, it is not replayed on fake2
    fake1 = FakeTTS(fake_audio_duration=1.0, fake_exception=APIConnectionError("connection lost"))
    fake2 = FakeTTS(fake_audio_duration=1.0)
    fallback_adapter = FallbackAdapterTester([fake1, fake2])
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)
    requests1 = fake1.stream_ch if streamed else fake1.synthesize_ch
    requests2 = fake2.stream_ch if streamed else fake2.synthesize_ch

    await _request(fallback_adapter, streamed=streamed)

    assert [e.recoverable for e in errors] == [True]
    assert isinstance(errors[0].error, APIConnectionError)
    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    with pytest.raises(ChanEmpty):
        requests2.recv_nowait()  # no replay on fake2

    # the probe starts now, not when a later request skips fake1
    assert await asyncio.wait_for(requests1.recv(), 1.0)  # the request fake1 failed
    assert await asyncio.wait_for(requests1.recv(), 1.0)  # its recovery probe
    status = fallback_adapter._status[0]
    assert status.recovering_stream_task if streamed else status.recovering_synthesize_task

    await fallback_adapter.aclose()


@pytest.mark.parametrize("streamed", [False, True])
async def test_failure_after_audio_ends_the_session_reply_without_closing(streamed: bool) -> None:
    # the primary fails after its audio started: apps see one recoverable error, and it does
    # not count toward closing the session. One sentence, so a non-streaming instance (wrapped
    # sentence by sentence) makes one request
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "Where is my order?")
    actions.add_llm("Your order has shipped.")

    fake = FakeTTS(fake_audio_duration=1.0, fake_exception=APIConnectionError("connection lost"))
    backup = FakeTTS(fake_audio_duration=1.0)
    if not streamed:
        _set_non_streaming(fake)
        _set_non_streaming(backup)
    fallback_adapter = FallbackAdapter([fake, backup])

    session = create_session(actions)
    session_errors: list[Any] = []
    session.on("error", session_errors.append)
    closes: list[Any] = []
    session.on("close", closes.append)

    agent = Agent(instructions="You are a helpful assistant.", tts=fallback_adapter)
    await asyncio.wait_for(run_session(session, agent), timeout=60)
    await fallback_adapter.aclose()

    assert [(e.error.type, e.error.recoverable) for e in session_errors] == [("tts_error", True)]
    assert [c.error for c in closes] == [None]  # only the close that run_session asks for


@pytest.mark.parametrize("streamed", [False, True])
async def test_failure_after_audio_is_unrecoverable_without_another_instance(
    streamed: bool,
) -> None:
    # nothing can serve the next utterance but the instance that just failed, as for a
    # realtime model with no fallback left
    fake = FakeTTS(fake_audio_duration=1.0, fake_exception=APIConnectionError("connection lost"))
    fallback_adapter = FallbackAdapterTester([fake])
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)

    await _request(fallback_adapter, streamed=streamed)

    assert [e.recoverable for e in errors] == [False]
    assert isinstance(errors[0].error, APIConnectionError)

    await fallback_adapter.aclose()


async def _all_instances_failed(*fakes: FakeTTS, streamed: bool) -> FallbackAdapterTester:
    # every instance is marked unavailable by a request that none of them can serve; probes are
    # then switched off (aclose) so only requests change what is available
    fallback_adapter = FallbackAdapterTester(list(fakes))
    with pytest.raises(APIConnectionError):
        await _request(fallback_adapter, streamed=streamed)
    for fake in fakes:
        assert not fallback_adapter.availability_changed_ch(fake).recv_nowait().available
    await fallback_adapter.aclose()
    return fallback_adapter


@pytest.mark.parametrize("streamed", [False, True])
async def test_secondary_that_serves_is_marked_available(streamed: bool) -> None:
    fake1 = _CountingTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = _CountingTTS(fake_exception=APIConnectionError("fake2 failed"))
    fallback_adapter = await _all_instances_failed(fake1, fake2, streamed=streamed)
    fake2.update_options(fake_exception=None, fake_audio_duration=1.0)

    # every instance is tried again, fake1 first, and fake2 serves
    await _request(fallback_adapter, streamed=streamed)

    ev = fallback_adapter.availability_changed_ch(fake2).recv_nowait()
    assert ev.tts is fake2 and ev.available
    assert [status.available for status in fallback_adapter._status] == [False, True]

    # fake1 is skipped and fake2, which works, is tried first
    attempts1 = fake1.user_attempts
    await _request(fallback_adapter, streamed=streamed)
    assert fake1.user_attempts == attempts1


@pytest.mark.parametrize("streamed", [False, True])
async def test_primary_that_serves_is_marked_available(streamed: bool) -> None:
    fake1 = _CountingTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = _CountingTTS(fake_exception=APIConnectionError("fake2 failed"))
    fallback_adapter = await _all_instances_failed(fake1, fake2, streamed=streamed)
    fake1.update_options(fake_exception=None, fake_audio_duration=1.0)

    await _request(fallback_adapter, streamed=streamed)

    ev = fallback_adapter.availability_changed_ch(fake1).recv_nowait()
    assert ev.tts is fake1 and ev.available
    assert [status.available for status in fallback_adapter._status] == [True, False]


@pytest.mark.parametrize("streamed", [False, True])
async def test_unavailable_secondary_serves_when_the_primary_alone_fails(streamed: bool) -> None:
    # after a total outage the primary serves first and is marked available, which leaves the
    # secondary unavailable although nothing is wrong with it. When the primary then fails
    # alone, the request goes to the secondary as a last resort instead of failing
    calls: list[str] = []
    fake1 = _CountingTTS(name="fake1", calls=calls, fake_exception=APIConnectionError("1 down"))
    fake2 = _CountingTTS(name="fake2", calls=calls, fake_exception=APIConnectionError("2 down"))
    fallback_adapter = await _all_instances_failed(fake1, fake2, streamed=streamed)
    fake1.update_options(fake_exception=None, fake_audio_duration=1.0)
    fake2.update_options(fake_exception=None, fake_audio_duration=1.0)
    await _request(fallback_adapter, streamed=streamed)  # fake1 serves
    assert [status.available for status in fallback_adapter._status] == [True, False]

    fake1.update_options(
        fake_exception=APIConnectionError("1 down again"), fake_audio_duration=None
    )
    errors: list[TTSError] = []
    fallback_adapter.on("error", errors.append)
    calls.clear()

    await _request(fallback_adapter, streamed=streamed)

    assert calls == ["fake1", "fake2"]  # fake2 is tried after fake1 failed, and serves
    assert errors == []
    assert [status.available for status in fallback_adapter._status] == [False, True]
    ev = fallback_adapter.availability_changed_ch(fake2).recv_nowait()
    assert ev.tts is fake2 and ev.available


@pytest.mark.parametrize("text", ["", " "])
async def test_request_without_speakable_text_does_not_restore_an_instance(text: str) -> None:
    # a non-streaming instance is wrapped in a StreamAdapter, which makes no provider call for
    # a request with no text: nothing shows the instance works
    fake1 = _set_non_streaming(_CountingTTS(fake_exception=APIConnectionError("1 down")))
    fake2 = _CountingTTS(fake_exception=APIConnectionError("2 down"))
    fallback_adapter = await _all_instances_failed(fake1, fake2, streamed=True)
    fake2.update_options(fake_exception=None, fake_audio_duration=1.0)
    attempts = fake1.user_attempts

    await _request(fallback_adapter, streamed=True, text=text)

    assert fake1.user_attempts == attempts  # no provider call
    assert [status.available for status in fallback_adapter._status] == [False, False]
    for fake in (fake1, fake2):
        with pytest.raises(ChanEmpty):
            fallback_adapter.availability_changed_ch(fake).recv_nowait()

    # a request with text still reaches fake2 after fake1 fails, and restores fake2 only
    await _request(fallback_adapter, streamed=True)
    assert [status.available for status in fallback_adapter._status] == [False, True]


class _GatedProbeTTS(FakeTTS):
    """FakeTTS whose recovery probes block until `gate` is set, so a probe stays in flight for
    exactly as long as the test needs. Requests for users are not gated."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate = asyncio.Event()

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        if not _in_recovery_probe.get():
            return super().stream(conn_options=conn_options)

        gate = self.gate

        class _Blocking(FakeSynthesizeStream):
            async def _run(self, output_emitter: AudioEmitter) -> None:
                await gate.wait()
                await super()._run(output_emitter)

        stream = _Blocking(tts=self, conn_options=conn_options)
        self._stream_ch.send_nowait(stream)
        return stream


async def test_instance_that_serves_while_its_probe_runs_is_reported_once() -> None:
    # unlike the tests above, the adapter is open: the probe the failed request started is still
    # running when the instance serves, then finishes
    fake = _GatedProbeTTS(fake_exception=APIConnectionError("down"))
    fallback_adapter = FallbackAdapterTester([fake], max_retry_per_tts=0)
    with pytest.raises(APIConnectionError):
        await _request(fallback_adapter, streamed=True)
    assert not fallback_adapter.availability_changed_ch(fake).recv_nowait().available
    assert await asyncio.wait_for(fake.stream_ch.recv(), 1.0)  # the failed request
    assert await asyncio.wait_for(fake.stream_ch.recv(), 1.0)  # the probe, parked on the gate
    status = fallback_adapter._status[0]
    probe = status.recovering_stream_task
    assert probe is not None and not probe.done()

    fake.update_options(fake_exception=None, fake_audio_duration=1.0)
    await _request(fallback_adapter, streamed=True)  # the instance serves while marked unavailable

    assert status.available
    assert fallback_adapter.availability_changed_ch(fake).recv_nowait().available
    assert not probe.done()

    fake.gate.set()
    await asyncio.wait_for(probe, 1.0)  # the probe succeeds, and finds nothing to change
    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake).recv_nowait()

    await fallback_adapter.aclose()


async def _probe_of(fallback_adapter: FallbackAdapter, index: int, *, streamed: bool) -> None:
    """Wait for the recovery probe the last request started on an instance."""
    status = fallback_adapter._status[index]
    probe = status.recovering_stream_task if streamed else status.recovering_synthesize_task
    assert probe is not None
    await asyncio.wait_for(probe, 1.0)


@pytest.mark.parametrize("streamed", [False, True])
async def test_probe_without_audio_does_not_restore_an_instance(streamed: bool) -> None:
    # the adapter stays open: the probe a request starts for a skipped instance really runs.
    # The request text is blank, so the probe makes no provider call (a non-streaming
    # instance is wrapped in a StreamAdapter) or gets no audio, and shows nothing about the
    # instance. Only audio restores it
    fake1 = FakeTTS(fake_exception=APIConnectionError("1 down"))
    if streamed:
        _set_non_streaming(fake1)
    else:
        fake1.update_options(fake_exception=None)  # the probe gets no audio and no error
    fake2 = FakeTTS(fake_audio_duration=1.0)
    fallback_adapter = FallbackAdapterTester([fake1, fake2])
    fallback_adapter._status[0].available = False

    await _request(fallback_adapter, streamed=streamed, text=" ")  # fake1 is skipped, fake2 serves
    await _probe_of(fallback_adapter, 0, streamed=streamed)

    assert not fallback_adapter._status[0].available
    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake1).recv_nowait()

    # a probe for real text gets audio and restores it
    fake1.update_options(fake_exception=None, fake_audio_duration=1.0)
    await _request(fallback_adapter, streamed=streamed)
    await _probe_of(fallback_adapter, 0, streamed=streamed)
    assert fallback_adapter._status[0].available
    assert fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    await fallback_adapter.aclose()


async def test_chunked_request_without_audio_does_not_restore_an_instance() -> None:
    # a chunked request with blank text gets no audio and no error (non-blank text with no
    # audio is an error). The instance served nothing, so it stays unavailable
    fake = FakeTTS()
    fallback_adapter = FallbackAdapterTester([fake])
    fallback_adapter._status[0].available = False

    await _request(fallback_adapter, streamed=False, text=" ")

    assert not fallback_adapter._status[0].available
    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake).recv_nowait()

    fake.update_options(fake_audio_duration=1.0)
    await _request(fallback_adapter, streamed=False)  # with audio it does
    assert fallback_adapter._status[0].available

    await fallback_adapter.aclose()


class _HookTTS(FakeTTS):
    """FakeTTS that calls `hook` when a request for a user reaches it."""

    def __init__(self, hook: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hook = hook

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        self._hook()
        return super().synthesize(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        self._hook()
        return super().stream(conn_options=conn_options)


@pytest.mark.parametrize("streamed", [False, True])
async def test_instance_restored_during_a_request_is_not_probed(streamed: bool) -> None:
    # fake1 is skipped as unavailable. While fake2 serves, fake1 is restored (by another
    # request or probe). When the request ends, it must not start a probe for the available fake1
    fake1 = FakeTTS(fake_audio_duration=1.0)
    fallback_adapter: FallbackAdapter

    def restore_fake1() -> None:
        fallback_adapter._set_available(fake1, fallback_adapter._status[0], True)

    fake2 = _HookTTS(restore_fake1, fake_audio_duration=1.0)
    fallback_adapter = FallbackAdapterTester([fake1, fake2])
    fallback_adapter._status[0].available = False

    await _request(fallback_adapter, streamed=streamed)

    status = fallback_adapter._status[0]
    assert status.available
    assert status.recovering_stream_task is None
    assert status.recovering_synthesize_task is None
    requests1 = fake1.stream_ch if streamed else fake1.synthesize_ch
    with pytest.raises(ChanEmpty):
        requests1.recv_nowait()  # fake1 never saw a request, probe or otherwise

    await fallback_adapter.aclose()


class _OverlapTTS(FakeTTS):
    """FakeTTS that counts its requests in flight, probes included, and keeps the most there
    were at once. Each request takes `hold` seconds."""

    def __init__(self, *, hold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hold = hold
        self.active = 0
        self.max_active = 0
        self.requests = 0

    async def _hold_request(self) -> None:
        self.requests += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self._hold)
        finally:
            self.active -= 1

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        outer = self

        class _Held(FakeChunkedStream):
            async def _run(self, output_emitter: AudioEmitter) -> None:
                await outer._hold_request()
                await super()._run(output_emitter)

        stream = _Held(tts=self, input_text=text, conn_options=conn_options)
        self._synthesize_ch.send_nowait(stream)
        return stream

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeSynthesizeStream:
        outer = self

        class _Held(FakeSynthesizeStream):
            async def _run(self, output_emitter: AudioEmitter) -> None:
                await outer._hold_request()
                await super()._run(output_emitter)

        stream = _Held(tts=self, conn_options=conn_options)
        self._stream_ch.send_nowait(stream)
        return stream


@pytest.mark.parametrize("streamed", [False, True])
async def test_probe_does_not_run_beside_the_last_resort_request(streamed: bool) -> None:
    # fake1 is skipped as unavailable and tried last, after fake2 fails. Its probe starts once
    # the request is over, not at the skip: the probes are not switched off here, so a probe
    # started at the skip would run while fake2 is still failing and overlap the last-resort
    # request on fake1
    fake1 = _OverlapTTS(hold=0.1, fake_exception=APIConnectionError("1 down"))
    fake2 = FakeTTS(fake_timeout=0.05, fake_exception=APIConnectionError("2 down"))
    fallback_adapter = FallbackAdapterTester([fake1, fake2], max_retry_per_tts=0)
    fallback_adapter._status[0].available = False

    with pytest.raises(APIConnectionError):
        await _request(fallback_adapter, streamed=streamed)

    await _probe_of(fallback_adapter, 0, streamed=streamed)

    assert fake1.requests == 2  # the last-resort request, then the probe
    assert fake1.max_active == 1  # never both at once

    await fallback_adapter.aclose()
