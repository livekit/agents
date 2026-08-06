from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIError, utils
from livekit.agents.tts import TTS, AvailabilityChangedEvent, FallbackAdapter
from livekit.agents.tts.tts import SynthesizedAudio, SynthesizeStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, USERDATA_TTS_STARTED_TIME
from livekit.agents.utils.aio.channel import ChanEmpty

from .fake_tts import FakeSynthesizeStream, FakeTTS

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
    # a provider marked unavailable is skipped on later streamed requests, so
    # nothing has awaited by the time recovery is considered and the probe used
    # to be dropped for want of text - leaving the process pinned to its
    # fallback for good after one transient outage (#6678)
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
