from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import VAD, APIError, VADCapabilities, VADStream
from livekit.agents.inference import VAD as InferenceVAD

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
async def test_audio_beyond_the_stream_bounds_is_refused_and_not_kept() -> None:
    vad = StalledVAD()
    stream = vad.stream()
    for _ in range(256):
        stream.push_frame(frame())
    with pytest.raises(APIError, match="capacity"):
        stream.push_frame(frame())
    with pytest.raises(APIError, match="byte limit"):
        stream.push_frame(frame(96_001))

    vad.reading.set()

    async def drained() -> None:
        while len(vad.received) < 256:
            await asyncio.sleep(0)

    await asyncio.wait_for(drained(), 1)
    stream.end_input()
    await asyncio.wait_for(stream._task, 1)
    assert len(vad.received) == 256


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
