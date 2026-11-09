"""SDK media helpers: audio frames and files, the audio stream decoder, and image encoding."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import aiofiles

from livekit import rtc

from .. import _core
from .._generated import public_utils_media as _public
from .._generated.public_utils_media import *  # noqa: F403
from .aio import cancel_and_wait


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


async def encode(frame: rtc.VideoFrame, options: EncodeOptions) -> bytes:
    """Encode a frame on an admitted worker, keeping the caller's event loop serviceable."""
    resize = options.resize_options
    return await _core.encode_image(
        frame,
        format=options.format,
        quality=options.quality,
        resize_width=resize.width if resize else None,
        resize_height=resize.height if resize else None,
        resize_strategy=resize.strategy if resize else None,
    )


__all__ = ["encode"]
__all__ += _public.__all__
