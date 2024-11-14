import asyncio

import pytest
from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.tts import TTS, AvailabilityChangedEvent, FallbackAdapter
from livekit.agents.utils.aio.channel import ChanEmpty

from .fake_tts import FakeTTS


class FallbackAdapterTester(FallbackAdapter):
    def __init__(
        self,
        tts: list[TTS],
        *,
        timeout: float = 7.5,
        no_fallback_after_audio_duration: float = 3.0,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            tts,
            timeout=timeout,
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

        fake1.synthesize_ch.recv_nowait()
        fake2.synthesize_ch.recv_nowait()

        assert rtc.combine_audio_frames(frames).duration == 5.0

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    fake2.update_options(fake_audio_duration=0.0)

    with pytest.raises(APIConnectionError):
        async with fallback_adapter.synthesize("hello test") as stream:
            async for _ in stream:
                pass

    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

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

        fake1.stream_ch.recv_nowait()
        fake2.stream_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    await fallback_adapter.aclose()


async def test_tts_recover() -> None:
    fake1 = FakeTTS(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeTTS(
        fake_exception=APIConnectionError("fake2 failed"), fake_connection_time=0.5
    )

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        async for _ in fallback_adapter.synthesize("hello test"):
            pass

        fake1.synthesize_ch.recv_nowait()
        fake2.synthesize_ch.recv_nowait()

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

    fake1.synthesize_ch.recv_nowait()
    fake2.synthesize_ch.recv_nowait()

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

        fake1.synthesize_ch.recv_nowait()
        fake2.synthesize_ch.recv_nowait()

        assert (
            not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
        )

        combined_frame = rtc.combine_audio_frames(frames)
        assert combined_frame.duration == 5.0
        assert combined_frame.sample_rate == 48000

    await fallback_adapter.aclose()
