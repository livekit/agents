from __future__ import annotations

import ctypes
from typing import List, Union

from livekit import rtc

from ..log import logger

# deprecated aliases
AudioBuffer = Union[List[rtc.AudioFrame], rtc.AudioFrame]

combine_frames = rtc.combine_audio_frames
merge_frames = rtc.combine_audio_frames


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
        if len(self._buf) == 0:
            return []

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
