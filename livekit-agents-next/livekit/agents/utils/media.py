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
    """Decode an audio file and yield its frames.

    Args:
        file_path: The path to the audio file.
        sample_rate: Desired sample rate.
        num_channels: Number of channels (1 for mono, 2 for stereo).
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
            # no EOF after a cancel: the decoder would report the truncated input as invalid audio
            if not aborted:
                decoder.end_input()

    reader_task = asyncio.create_task(file_reader())

    try:
        async for frame in decoder:
            yield frame
    finally:
        await cancel_and_wait(reader_task)
        await decoder.aclose()

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
