from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import VAD, APIError, VADCapabilities, VADEvent, VADEventType, VADStream
from livekit.agents.inference import VAD as InferenceVAD
from livekit.agents.utils import aio

pytestmark = pytest.mark.unit


class StalledVAD(VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.1))
        self.reading = asyncio.Event()
        self.received: list[rtc.AudioFrame] = []

    def stream(self) -> VADStream:
        return StalledStream(self)


class StalledStream(VADStream):
    async def _main_task(self) -> None:
        assert isinstance(self._vad, StalledVAD)
        await self._vad.reading.wait()
        async for item in self.input:
            if isinstance(item, rtc.AudioFrame):
                self._vad.received.append(item)


def frame(samples: int = 160) -> rtc.AudioFrame:
    return rtc.AudioFrame(bytes(2 * samples), 16000, 1, samples)


@pytest.mark.asyncio
async def test_a_full_input_refuses_a_direct_push_and_keeps_what_it_holds() -> None:
    vad = StalledVAD()
    stream = vad.stream()
    for _ in range(256):
        stream.push_frame(frame())
    with pytest.raises(aio.ChanFull):
        stream.push_frame(frame())

    vad.reading.set()

    async def drained() -> None:
        while len(vad.received) < 256:
            await asyncio.sleep(0)

    await asyncio.wait_for(drained(), 1)
    stream.end_input()
    await asyncio.wait_for(stream._task, 1)
    assert len(vad.received) == 256


@pytest.mark.asyncio
async def test_the_core_push_waits_for_a_full_input_instead_of_failing() -> None:
    vad = StalledVAD()
    stream = vad.stream()
    pushing = asyncio.create_task(_core_push_all(stream, 300))
    await asyncio.sleep(0.05)
    assert not pushing.done(), "the core's pump waits, its queue keeps the audio"

    vad.reading.set()
    await asyncio.wait_for(pushing, 1)
    await asyncio.wait_for(stream._core_end_input(), 1)
    await asyncio.wait_for(stream._task, 1)
    assert len(vad.received) == 300


async def _core_push_all(stream: VADStream, frames: int) -> None:
    for _ in range(frames):
        await stream._core_push(frame())


@pytest.mark.asyncio
async def test_close_completes_with_a_full_input() -> None:
    stream = StalledVAD().stream()
    for _ in range(256):
        stream.push_frame(frame())
    await asyncio.wait_for(stream.aclose(), 1)


class RefusingNative:
    """The core's VAD stream at capacity: it refuses the next push and ends with its input."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ended = asyncio.Event()

    def stream(self) -> RefusingNative:
        return self

    def push(self, frame: rtc.AudioFrame) -> None:
        if self.capacity == 0:
            raise APIError("VAD input capacity exhausted", retryable=False)
        self.capacity -= 1

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        self.ended.set()

    async def recv(self) -> None:
        await self.ended.wait()

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_inference_vad_refused_audio_fails_the_stream() -> None:
    vad = object.__new__(InferenceVAD)
    VAD.__init__(vad, capabilities=VADCapabilities(update_interval=0.032))
    vad._native = RefusingNative(capacity=4)
    async with vad.stream() as stream:
        for _ in range(8):
            stream.push_frame(frame())
        with pytest.raises(APIError, match="capacity"):
            async with asyncio.timeout(5):
                async for _ in stream:
                    pass


class EchoVAD(VAD):
    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.1))

    def stream(self) -> VADStream:
        return EchoStream(self)


class EchoStream(VADStream):
    """One event per frame, several frames' worth from one large frame."""

    async def _main_task(self) -> None:
        async for item in self.input:
            if isinstance(item, rtc.AudioFrame):
                for _ in range(item.samples_per_channel // 160):
                    self.send(
                        VADEvent(
                            type=VADEventType.INFERENCE_DONE,
                            samples_index=0,
                            timestamp=0,
                            speech_duration=0,
                            silence_duration=0,
                        )
                    )


@pytest.mark.asyncio
async def test_a_reader_behind_on_events_holds_the_core_push() -> None:
    stream = EchoVAD().stream()
    await stream._core_push(frame(160 * 100))
    pushing = asyncio.create_task(_core_push_all(stream, 512))
    await asyncio.sleep(0.05)
    assert not pushing.done(), "pending events stop the detector, and the full input the pump"
    # nothing the detector produced was dropped
    assert len([await anext(stream) for _ in range(100)]) == 100
    await stream.aclose()
    pushing.cancel()
