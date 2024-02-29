import asyncio
from typing import Union, List
from livekit import rtc

AudioBuffer = Union[List[rtc.AudioFrame], rtc.AudioFrame]


def merge_frames(buffer: AudioBuffer) -> rtc.AudioFrame:
    """
    Merges one or more AudioFrames into a single one
    Args:
        buffer: either a rtc.AudioFrame or a list of rtc.AudioFrame
    """
    if isinstance(buffer, list):
        # merge all frames into one
        if len(buffer) == 0:
            raise ValueError("buffer is empty")

        sample_rate = buffer[0].sample_rate
        num_channels = buffer[0].num_channels
        samples_per_channel = 0
        data = b""
        for frame in buffer:
            if frame.sample_rate != sample_rate:
                raise ValueError("sample rate mismatch")

            if frame.num_channels != num_channels:
                raise ValueError("channel count mismatch")

            data += frame.data
            samples_per_channel += frame.samples_per_channel

        return rtc.AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

    return buffer


class AsyncIterableQueue:
    def __init__(self):
        self._queue = asyncio.Queue()

    def put_nowait(self, item):
        self._queue.put_nowait(item)

    async def put(self, item):
        await self._queue.put(item)

    def close(self):
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item
