from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import aiofiles
import numpy as np
from numpy.typing import DTypeLike

from livekit import rtc

from .. import _core
from .aio import cancel_and_wait

# the buffer type and frame helpers plugins use
AudioBuffer = list[rtc.AudioFrame] | rtc.AudioFrame

combine_frames = rtc.combine_audio_frames
merge_frames = rtc.combine_audio_frames


def silence_frame(duration: float, sample_rate: int, num_channels: int = 1) -> rtc.AudioFrame:
    """Create a zeroed ``rtc.AudioFrame`` of the given duration and format."""
    samples = int(duration * sample_rate)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples * num_channels,
        num_channels=num_channels,
        samples_per_channel=samples,
        sample_rate=sample_rate,
    )


def calculate_audio_duration(frames: AudioBuffer) -> float:
    """
    Calculate the total duration of audio frames.

    This function computes the total duration of audio frames in seconds.
    It accepts either a list of `rtc.AudioFrame` objects or a single `rtc.AudioFrame` object.

    Parameters:
    - frames (AudioBuffer): A list of `rtc.AudioFrame` instances or a single `rtc.AudioFrame` instance.

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


class AudioArrayBuffer:
    def __init__(self, *, buffer_size: int, dtype: DTypeLike = np.int16, sample_rate: int = 16000):
        """Create a fixed-size buffer for audio array data.

        Args:
            buffer_size: The size of the buffer in samples.
            dtype: The dtype of the buffer.
            sample_rate: The sample rate of the buffer.
        """
        self._buffer_size = buffer_size
        self._dtype = dtype
        self._buffer = np.zeros(buffer_size, dtype=dtype)
        self._start_idx = 0
        self._resampler: rtc.AudioResampler | None = None
        self._sample_rate = sample_rate

    def push_frame(self, frame: rtc.AudioFrame) -> int:
        """Push an audio frame to the buffer.

        Args:
            frame: The audio frame to push.

        Returns:
            The number of samples written to the buffer.

        Raises:
            ValueError: If the frame samples, after resampling, are greater than the buffer size.
        """
        frames: list[rtc.AudioFrame] = []
        if self._resampler is None and frame.sample_rate != self._sample_rate:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._sample_rate,
                num_channels=frame.num_channels,
                quality=rtc.AudioResamplerQuality.QUICK,
            )

        if self._resampler:
            if frame.sample_rate != self._resampler._input_rate:
                raise ValueError("frame sample rates are inconsistent")
            frames.extend(self._resampler.push(frame))
        else:
            frames.append(frame)

        if not frames:
            # the resampler holds short frames back until it can emit a full output frame
            return 0

        if (samples := sum(f.samples_per_channel for f in frames)) > self._buffer_size:
            detail = (
                f" after resampling {frame.sample_rate}Hz to {self._sample_rate}Hz"
                if self._resampler
                else ""
            )
            raise ValueError(
                f"frame samples ({samples}{detail}) are greater than "
                f"the buffer size ({self._buffer_size})"
            )

        frame = merge_frames(frames)

        if (shift_size := self._start_idx + frame.samples_per_channel - self._buffer_size) > 0:
            self.shift(shift_size)
        ptr = self._buffer[self._start_idx : self._start_idx + frame.samples_per_channel]
        if frame.num_channels > 1:
            arr_i16 = np.frombuffer(
                frame.data, dtype=np.int16, count=frame.samples_per_channel * frame.num_channels
            ).reshape(-1, frame.num_channels)
            ptr[:] = (np.sum(arr_i16, axis=1, dtype=np.int32) // frame.num_channels).astype(
                np.int16
            )
        else:
            ptr[:] = np.frombuffer(frame.data, dtype=np.int16, count=frame.samples_per_channel)
        self._start_idx += frame.samples_per_channel
        return frame.samples_per_channel

    def shift(self, size: int) -> None:
        """Shift the buffer to the left by the given size.

        Args:
            size: The size to shift the buffer by.
        """
        size = min(size, self._start_idx)
        self._buffer[: self._start_idx - size] = self._buffer[size : self._start_idx].copy()
        self._start_idx -= size

    def read(self) -> np.ndarray:
        return self._buffer[: self._start_idx].copy()

    def reset(self) -> None:
        self._start_idx = 0
        self._buffer.fill(0)
        self._resampler = None

    def __len__(self) -> int:
        return self._start_idx
