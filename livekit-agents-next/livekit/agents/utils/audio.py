from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import aiofiles

from livekit import rtc

from .. import _core
from .aio import cancel_and_wait


def silence_frame(duration: float, sample_rate: int, num_channels: int = 1) -> rtc.AudioFrame:
    """Create a zeroed ``rtc.AudioFrame`` of the given duration and format."""
    samples = int(duration * sample_rate)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples * num_channels,
        num_channels=num_channels,
        samples_per_channel=samples,
        sample_rate=sample_rate,
    )


def calculate_audio_duration(frames: list[rtc.AudioFrame] | rtc.AudioFrame) -> float:
    """
    Calculate the total duration of audio frames.

    This function computes the total duration of audio frames in seconds.
    It accepts either a list of `rtc.AudioFrame` objects or a single `rtc.AudioFrame` object.

    Parameters:
    - frames (list[rtc.AudioFrame] | rtc.AudioFrame): A list of `rtc.AudioFrame` instances or a single `rtc.AudioFrame` instance.

    Returns:
    - float: The total duration in seconds of all frames provided.
    """  # noqa: E501
    if isinstance(frames, list):
        return sum(frame.duration for frame in frames)
    else:
        return frames.duration


AudioByteStream = _core.AudioByteStream


async def audio_frames_from_file(
    file_path: str, sample_rate: int = 48000, num_channels: int = 1
) -> AsyncGenerator[rtc.AudioFrame, None]:
    """
    Decode the audio file into rtc.AudioFrame instances and yield them as an async iterable.
    Args:
        file_path (str): The path to the audio file.
        sample_rate (int, optional): Desired sample rate. Defaults to 48000.
        num_channels (int, optional): Number of channels (1 for mono, 2 for stereo). Defaults to 1.
    Returns:
        AsyncIterable[rtc.AudioFrame]: An async iterable that yields decoded AudioFrame
    """
    from .codecs import AudioStreamDecoder

    decoder = AudioStreamDecoder(sample_rate=sample_rate, num_channels=num_channels)

    async def file_reader() -> None:
        aborted = False
        try:
            async with aiofiles.open(file_path, mode="rb") as f:
                while True:
                    chunk = await f.read(4096)
                    if not chunk:
                        break

                    decoder.push(chunk)
        except asyncio.CancelledError:
            aborted = True
            raise
        finally:
            # a cancelled read leaves a truncated file, not an end of input: signalling EOF
            # would make the decoder report the abort as invalid audio. aclose() closes it.
            if not aborted:
                decoder.end_input()

    reader_task = asyncio.create_task(file_reader())

    try:
        async for frame in decoder:
            yield frame
    finally:
        await cancel_and_wait(reader_task)
        await decoder.aclose()

    # propagate file reader errors (e.g. FileNotFoundError for missing files)
    if reader_task.done() and not reader_task.cancelled():
        if exc := reader_task.exception():
            raise exc
