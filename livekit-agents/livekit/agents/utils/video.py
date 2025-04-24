from collections.abc import AsyncIterator
from time import perf_counter

from livekit import rtc


class VideoFPSSampler:
    def __init__(self, fps: float):
        self._fps = fps
        self._last_sample_time = 0

    def should_sample(self, event: rtc.VideoFrameEvent) -> bool:
        time_delta = 1.0 / self._fps
        current_time = perf_counter()

        if current_time - self._last_sample_time >= time_delta:
            self._last_sample_time = current_time
            return True

        return False


class SamplingVideoStream:
    def __init__(self, stream: AsyncIterator[rtc.VideoFrameEvent], sampler: VideoFPSSampler):
        self._stream = stream
        self._sampler = sampler

    def __aiter__(self):
        return self

    async def __anext__(self) -> rtc.VideoFrameEvent:
        while True:
            event = await self._stream.__anext__()
            if self._sampler.should_sample(event):
                return event
