from __future__ import annotations

import ctypes
from typing import List, Union

from livekit import rtc

from ..log import logger

AudioBuffer = Union[List[rtc.AudioFrame], rtc.AudioFrame]


def combine_frames(buffer: AudioBuffer) -> rtc.AudioFrame:
    """
    Combines one or more `rtc.AudioFrame` objects into a single `rtc.AudioFrame`.

    This function concatenates the audio data from multiple frames, ensuring that
    all frames have the same sample rate and number of channels. It efficiently
    merges the data by preallocating the necessary memory and copying the frame
    data without unnecessary reallocations.

    Args:
        buffer (AudioBuffer): A single `rtc.AudioFrame` or a list of `rtc.AudioFrame`
            objects to be combined.

    Returns:
        rtc.AudioFrame: A new `rtc.AudioFrame` containing the combined audio data.

    Raises:
        ValueError: If the buffer is empty.
        ValueError: If frames have differing sample rates.
        ValueError: If frames have differing numbers of channels.

    Example:
        >>> frame1 = rtc.AudioFrame(
        ...     data=b"\x01\x02", sample_rate=48000, num_channels=2, samples_per_channel=1
        ... )
        >>> frame2 = rtc.AudioFrame(
        ...     data=b"\x03\x04", sample_rate=48000, num_channels=2, samples_per_channel=1
        ... )
        >>> combined_frame = combine_frames([frame1, frame2])
        >>> combined_frame.data
        b'\x01\x02\x03\x04'
        >>> combined_frame.sample_rate
        48000
        >>> combined_frame.num_channels
        2
        >>> combined_frame.samples_per_channel
        2
    """
    if not isinstance(buffer, list):
        return buffer

    if not buffer:
        raise ValueError("buffer is empty")

    sample_rate = buffer[0].sample_rate
    num_channels = buffer[0].num_channels

    total_data_length = 0
    total_samples_per_channel = 0

    for frame in buffer:
        if frame.sample_rate != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: expected {sample_rate}, got {frame.sample_rate}"
            )

        if frame.num_channels != num_channels:
            raise ValueError(
                f"Channel count mismatch: expected {num_channels}, got {frame.num_channels}"
            )

        total_data_length += len(frame.data)
        total_samples_per_channel += frame.samples_per_channel

    data = bytearray(total_data_length)
    offset = 0
    for frame in buffer:
        frame_data = frame.data.cast("b")
        data[offset : offset + len(frame_data)] = frame_data
        offset += len(frame_data)

    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=total_samples_per_channel,
    )


merge_frames = combine_frames


class AudioByteStream:
    """
    Buffer and chunk audio byte data into fixed-size frames.

    This class is designed to handle incoming audio data in bytes,
    buffering it and producing audio frames of a consistent size.
    It is mainly used to easily chunk big or too small audio frames
    into a fixed size, helping to avoid processing very small frames
    (which can be inefficient) and very large frames (which can cause
    latency or processing delays). By normalizing frame sizes, it
    facilitates consistent and efficient audio data processing.
    """

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
    ) -> None:
        """
        Initialize an AudioByteStream instance.

        Parameters:
            sample_rate (int): The audio sample rate in Hz.
            num_channels (int): The number of audio channels.
            samples_per_channel (int, optional): The number of samples per channel in each frame.
                If None, defaults to `sample_rate // 10` (i.e., 100ms of audio data).

        The constructor sets up the internal buffer and calculates the size of each frame in bytes.
        The frame size is determined by the number of channels, samples per channel, and the size
        of each sample (assumed to be 16 bits or 2 bytes).
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if samples_per_channel is None:
            samples_per_channel = sample_rate // 10  # 100ms by default

        self._bytes_per_frame = (
            num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        )
        self._buf = bytearray()

    def push(self, data: bytes) -> list[rtc.AudioFrame]:
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
        while len(self._buf) >= self._bytes_per_frame:
            frame_data = self._buf[: self._bytes_per_frame]
            self._buf = self._buf[self._bytes_per_frame :]

            frames.append(
                rtc.AudioFrame(
                    data=frame_data,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=len(frame_data) // 2,
                )
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
        if len(self._buf) % (2 * self._num_channels) != 0:
            logger.warning("AudioByteStream: incomplete frame during flush, dropping")
            return []

        return [
            rtc.AudioFrame(
                data=self._buf,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=len(self._buf) // 2,
            )
        ]
