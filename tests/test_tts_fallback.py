from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.tts import TTS, AvailabilityChangedEvent, FallbackAdapter
from livekit.agents.tts.tts import SynthesizeStream
from livekit.agents.utils.aio.channel import ChanEmpty

from .fake_tts import FakeTTS


class FallbackAdapterTester(FallbackAdapter):
    def __init__(
        self,
        tts: list[TTS],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_tts: int = 1,  # only retry once by default
        no_fallback_after_audio_duration: float | None = 3.0,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            tts,
            attempt_timeout=attempt_timeout,
            max_retry_per_tts=max_retry_per_tts,
            no_fallback_after_audio_duration=no_fallback_after_audio_duration,
            sample_rate=sample_rate,
        )

        self.on("tts_availability_changed", self._on_tts_availability_changed)

        self._availability_changed_ch: dict[
            int, utils.aio.Chan[AvailabilityChangedEvent]
        ] = {id(t): utils.aio.Chan[AvailabilityChangedEvent]() for t in tts}

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

        assert rtc.combine_audio_frames(frames).duration == 5.0

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
        async with fallback_adapter.synthesize("hello test") as stream:
            async for _ in stream:
                pass

    # stream
    fake1.update_options(fake_audio_duration=5.0)

    async def _input_task(stream: SynthesizeStream):
        with contextlib.suppress(RuntimeError):
            stream.push_text("hello test")
            stream.flush()
            await asyncio.sleep(1.0)

            fake1.update_options(fake_timeout=0.5, fake_audio_duration=None)

            stream.push_text("hello test")
            stream.end_input()

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.stream() as stream:
            input_task = asyncio.create_task(_input_task(stream))

            segments = set()
            try:
                async for ev in stream:
                    segments.add(ev.segment_id)
            finally:
                await input_task

            assert len(segments) == 1

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


async def test_tts_recover() -> None:
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(fake_exception=APIConnectionError("fake2 failed"), fake_timeout=0.5)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        async for _ in fallback_adapter.synthesize("hello test"):
            pass

        assert fake1.synthesize_ch.recv_nowait()
        assert fake2.synthesize_ch.recv_nowait()

    fake2.update_options(fake_exception=None, fake_audio_duration=5.0)

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert (
        await asyncio.wait_for(
            fallback_adapter.availability_changed_ch(fake2).recv(), 1.0
        )
    ).available, "fake2 should have recovered"

    async for _ in fallback_adapter.synthesize("hello test"):
        pass

    assert fake1.synthesize_ch.recv_nowait()
    assert fake2.synthesize_ch.recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake1).recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake2).recv_nowait()

    await fallback_adapter.aclose()


async def test_audio_resampled() -> None:
    fake1 = FakeTTS(
        sample_rate=48000, fake_exception=APIConnectionError("fake1 failed")
    )
    fake2 = FakeTTS(fake_audio_duration=5.0, sample_rate=16000)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.synthesize("hello test") as stream:
        frames = []
        async for data in stream:
            frames.append(data.frame)

        assert fake1.synthesize_ch.recv_nowait()
        assert fake2.synthesize_ch.recv_nowait()

        assert (
            not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
        )

        combined_frame = rtc.combine_audio_frames(frames)
        assert combined_frame.duration == 5.0
        assert combined_frame.sample_rate == 48000

    assert await asyncio.wait_for(fake1.synthesize_ch.recv(), 1.0)

    async with fallback_adapter.stream() as stream:
        stream.push_text("hello test")
        stream.end_input()

        frames = []
        async for data in stream:
            frames.append(data.frame)

        print(frames)

        assert fake2.stream_ch.recv_nowait()

        combined_frame = rtc.combine_audio_frames(frames)
        assert combined_frame.duration == 5.0
        assert combined_frame.sample_rate == 48000

    await fallback_adapter.aclose()


async def test_timeout():
    fake1 = FakeTTS(fake_timeout=0.5, sample_rate=48000)
    fake2 = FakeTTS(fake_timeout=0.5, sample_rate=48000)

    fallback_adapter = FallbackAdapterTester([fake1, fake2], attempt_timeout=0.1)

    with pytest.raises(APIConnectionError):
        async for _ in fallback_adapter.synthesize("hello test"):
            pass

    assert fake1.synthesize_ch.recv_nowait()
    assert fake2.synthesize_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert await asyncio.wait_for(fake1.synthesize_ch.recv(), 1.0)
    assert await asyncio.wait_for(fake2.synthesize_ch.recv(), 1.0)

    # stream
    with pytest.raises(APIConnectionError):
        async with fallback_adapter.stream() as stream:
            stream.end_input()
            async for _ in stream:
                pass

    assert fake1.stream_ch.recv_nowait()
    assert fake2.stream_ch.recv_nowait()

    assert await asyncio.wait_for(fake1.stream_ch.recv(), 1.0)
    assert await asyncio.wait_for(fake2.stream_ch.recv(), 1.0)

    await fallback_adapter.aclose()

    # consecutive push must not timeout
    fake1.update_options(fake_timeout=None, fake_audio_duration=5.0)
    fallback_adapter = FallbackAdapterTester([fake1], attempt_timeout=0.25)

    async def _input_task1(stream: SynthesizeStream):
        stream.push_text("hello world")
        stream.flush()
        await asyncio.sleep(1.0)

        stream.push_text("bye world")
        stream.end_input()

    async with fallback_adapter.stream() as stream:
        input_task = asyncio.create_task(_input_task1(stream))

        segments = set()
        final_count = 0
        async for ev in stream:
            segments.add(ev.segment_id)
            if ev.is_final:
                final_count += 1

        assert len(segments) == 2
        assert final_count == 2
        await input_task

    async def _input_task2(stream: SynthesizeStream):
        with contextlib.suppress(RuntimeError):
            stream.push_text("hello test")
            stream.flush()
            await asyncio.sleep(1.0)

            fake1.update_options(fake_timeout=0.5, fake_audio_duration=None)

            stream.push_text("hello test")
            stream.flush()
            await asyncio.sleep(1.0)

            stream.end_input()

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.stream() as stream:
            input_task = asyncio.create_task(_input_task2(stream))

            try:
                async for ev in stream:
                    pass
            finally:
                await input_task

    await fallback_adapter.aclose()
