from __future__ import annotations

import ctypes

from livekit import rtc

from ..log import logger


class AudioByteStream:
    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if samples_per_channel is None:
            samples_per_channel = sample_rate // 50  # 20ms by default

        self._bytes_per_frame = (
            num_channels * samples_per_channel * ctypes.sizeof(ctypes.c_int16)
        )
        self._buf = bytearray()

    def write(self, data: bytes) -> list[rtc.AudioFrame]:
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

    def flush(self) -> list[rtc.AudioFrame]:
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
