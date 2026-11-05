"""SDK media helpers: audio frames and files, the audio stream decoder, and image encoding."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

import aiofiles

from livekit import rtc
from livekit.agents._generated.image_options import (
    EncodeOptions as EncodeOptions,
    ResizeOptions as ResizeOptions,
)

from .. import _core
from ..log import logger
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


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream. The decoding is the core's (``_core.AudioDecoder``):
    its frames come out as soon as they are decoded.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = 48000,
        num_channels: int | None = 1,
        format: str | None = None,
    ):
        """
        Args:
            sample_rate: The output rate; the source's when ``None``.
            num_channels: The output channel count.
            format: The input's media type (``"audio/mpeg"``); probed when ``None``.
        """
        self._decoder: _core.AudioDecoder | None = None
        self._ended = False
        try:
            self._decoder = _core.AudioDecoder(
                format.lower() if format else "",
                sample_rate=sample_rate,
                num_channels=num_channels or 1,
            )
        except RuntimeError:
            logger.exception("error decoding audio")

    def push(self, chunk: bytes) -> None:
        if self._decoder is None or self._ended:
            return
        try:
            self._decoder.push(chunk)
        except RuntimeError:
            logger.exception("error decoding audio")
            self.end_input()

    def end_input(self) -> None:
        self._ended = True
        if self._decoder is not None:
            self._decoder.end_input()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame: rtc.AudioFrame | None = None
        if self._decoder is not None:
            try:
                frame = await self._decoder.next()
            except RuntimeError:
                logger.exception("error decoding audio")
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def aclose(self) -> None:
        decoder, self._decoder = self._decoder, None
        if decoder is not None:
            await decoder.aclose()


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


class _FrameReader:
    """How the core reads a host frame it encodes: as RGBA, converted from any other type."""

    def _core_rgba(self, frame: rtc.VideoFrame) -> rtc.VideoFrame:
        if frame.type == rtc.VideoBufferType.RGBA:
            return frame
        return frame.convert(rtc.VideoBufferType.RGBA)


_core._core_install_frame_reader(_FrameReader())


__all__ = [
    "AudioStreamDecoder",
    "EncodeOptions",
    "ResizeOptions",
    "encode",
]
