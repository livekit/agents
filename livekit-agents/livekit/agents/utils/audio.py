from __future__ import annotations

import asyncio
import ctypes
from collections.abc import AsyncGenerator

import aiofiles
import numpy as np
from numpy.typing import DTypeLike

from livekit import rtc

from ..log import logger
from .aio.utils import cancel_and_wait

# deprecated aliases
AudioBuffer = list[rtc.AudioFrame] | rtc.AudioFrame

combine_frames = rtc.combine_audio_frames
merge_frames = rtc.combine_audio_frames


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


class AudioByteStream:
    """Buffer and chunk audio byte data into fixed-size frames.

    Accepts variable-sized byte chunks (e.g. from a network stream or file) and
    emits consistently-sized ``rtc.AudioFrame`` objects.

    Two modes of operation:

    * **Fixed** (``progressive=False``, the default): every emitted frame is
      exactly ``samples_per_channel`` samples long.
    * **Progressive** (``progressive=True``): the *first* emitted frame is only
      20 ms of audio.  Each subsequent frame doubles in size until
      ``samples_per_channel`` is reached.  This minimises time-to-first-audio
      while giving the pipeline a brief warm-up before reaching full frame
      sizes.

    Example with ``sample_rate=16000, samples_per_channel=3200`` (200 ms) and
    ``progressive=True``::

        Frame 1:  320 samples  ( 20 ms)
        Frame 2:  640 samples  ( 40 ms)
        Frame 3: 1280 samples  ( 80 ms)
        Frame 4: 2560 samples  (160 ms)
        Frame 5: 3200 samples  (200 ms)   ← target reached
        Frame 6: 3200 samples  (200 ms)
        ...
    """

    _MIN_PROGRESSIVE_MS = 20

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
        progressive: bool = False,
    ) -> None:
        """
        Args:
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            samples_per_channel: Target samples per channel in each emitted frame.
                Defaults to ``sample_rate // 10`` (100 ms).
            progressive: When *True*, start with a small 20 ms frame and double
                the frame size on each subsequent emission until
                ``samples_per_channel`` is reached.
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if samples_per_channel is None:
            samples_per_channel = sample_rate // 10  # 100ms by default

        self._bytes_per_sample = num_channels * ctypes.sizeof(ctypes.c_int16)
        self._target_bytes_per_frame = samples_per_channel * self._bytes_per_sample
        self._buf = bytearray()

        if progressive:
            min_samples = sample_rate * self._MIN_PROGRESSIVE_MS // 1000
            self._initial_bytes_per_frame = min(
                min_samples * self._bytes_per_sample, self._target_bytes_per_frame
            )
        else:
            self._initial_bytes_per_frame = self._target_bytes_per_frame
        self._current_bytes_per_frame = self._initial_bytes_per_frame

    def push(self, data: bytes | memoryview) -> list[rtc.AudioFrame]:
        """
        Add audio data to the buffer and retrieve fixed-size frames.

        Parameters:
            data (bytes): The incoming audio data to buffer.

        Returns:
            list[rtc.AudioFrame]: A list of `AudioFrame` objects of fixed size.

        The method appends the incoming data to the internal buffer.
        While the buffer contains enough data to form complete frames,
        it extracts the data for each frame, creates an `AudioFrame` object,
        and appends it to the list of frames to return.

        This allows you to feed in variable-sized chunks of audio data
        (e.g., from a stream or file) and receive back a list of
        fixed-size audio frames ready for processing or transmission.
        """
        self._buf.extend(data)

        frames = []
        while len(self._buf) >= self._current_bytes_per_frame:
            frame_data = self._buf[: self._current_bytes_per_frame]
            del self._buf[: self._current_bytes_per_frame]

            frames.append(
                rtc.AudioFrame(
                    data=frame_data,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=len(frame_data) // self._bytes_per_sample,
                )
            )

            # progressively double toward the target frame size
            if self._current_bytes_per_frame < self._target_bytes_per_frame:
                self._current_bytes_per_frame = min(
                    self._current_bytes_per_frame * 2, self._target_bytes_per_frame
                )

        return frames

    write = push  # Alias for the push method.

    def flush(self) -> list[rtc.AudioFrame]:
        """
        Flush the buffer and retrieve any remaining audio data as a frame.

        Returns:
            list[rtc.AudioFrame]: A list containing any remaining `AudioFrame` objects.

        This method processes any remaining data in the buffer that does not
        fill a complete frame. If the remaining data forms a partial frame
        (i.e., its size is not a multiple of the expected sample size), a warning is
        logged and an empty list is returned. Otherwise, it returns the final
        `AudioFrame` containing the remaining data.

        Use this method when you have no more data to push and want to ensure
        that all buffered audio data has been processed.
        """
        if len(self._buf) == 0:
            return []

        if len(self._buf) % self._bytes_per_sample != 0:
            logger.warning("AudioByteStream: incomplete frame during flush, dropping")
            return []

        frames = [
            rtc.AudioFrame(
                data=self._buf.copy(),
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=len(self._buf) // self._bytes_per_sample,
            )
        ]
        self._buf.clear()
        return frames

    def clear(self) -> None:
        """Discard all buffered data and reset progressive frame sizing.

        After clearing, the next :meth:`push` will start from the initial
        (small) frame size again, ensuring low latency on the first frame
        after an interruption.
        """
        self._buf.clear()
        self._current_bytes_per_frame = self._initial_bytes_per_frame


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
        try:
            async with aiofiles.open(file_path, mode="rb") as f:
                while True:
                    chunk = await f.read(4096)
                    if not chunk:
                        break

                    decoder.push(chunk)
        finally:
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
            ValueError: If the frame samples are greater than the buffer size.
        """
        if frame.samples_per_channel > self._buffer_size:
            raise ValueError("frame samples are greater than the buffer size")

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

    def __len__(self) -> int:
        return self._start_idx
