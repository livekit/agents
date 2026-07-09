# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RNNoise noise-cancellation FrameProcessor for LiveKit Agents.

RNNoise operates on fixed 48kHz mono 480-sample (10ms) frames. Input frames can
arrive at any sample rate/cadence, so this module resamples to 48kHz, feeds
RNNoise exact 480-sample chunks, and resamples the denoised audio back to the
source rate -- returning, on every call, a frame with the same sample_rate,
num_channels and samples_per_channel as the input.
"""

from __future__ import annotations

import numpy as np

# pyrnnoise ships no stubs/py.typed marker upstream.
from pyrnnoise import RNNoise as _RNNoiseDenoiser  # type: ignore[import-untyped]

from livekit import rtc

_RNNOISE_SAMPLE_RATE = 48000
_RNNOISE_FRAME_SAMPLES = 480

_EMPTY_SAMPLES = np.empty(0, dtype=np.int16)


class RNNoiseFrameProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """FrameProcessor implementation for RNNoise noise cancellation.

    Example:
        ```python
        from livekit.agents import room_io
        from livekit.plugins import rnnoise

        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=rnnoise.RNNoise(),
                ),
            ),
        )
        ```
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._denoiser = _RNNoiseDenoiser(_RNNOISE_SAMPLE_RATE)

        self._source_rate: int | None = None
        self._num_channels: int | None = None
        self._in_resampler: rtc.AudioResampler | None = None
        self._out_resampler: rtc.AudioResampler | None = None

        # Samples awaiting denoising, always at 48kHz.
        self._in_buffer = _EMPTY_SAMPLES
        # Denoised samples awaiting resample back to the source rate.
        self._pending_denoised_48k = _EMPTY_SAMPLES
        # Samples at the source rate awaiting return to the caller.
        self._out_buffer = _EMPTY_SAMPLES

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        if frame.num_channels != 1:
            raise ValueError(
                "RNNoiseFrameProcessor v1 only supports mono audio, but "
                f"received a frame with {frame.num_channels} channels"
            )

        n = frame.samples_per_channel
        self._ensure_pipeline(frame.sample_rate, frame.num_channels)

        in_48k = self._resample_to_rnnoise(frame)
        self._in_buffer = np.concatenate([self._in_buffer, in_48k])
        self._denoise_complete_chunks()
        self._resample_denoised_to_source(frame.sample_rate)

        return self._take_output_frame(frame, n)

    def _close(self) -> None:
        self._denoiser = _RNNoiseDenoiser(_RNNOISE_SAMPLE_RATE)
        self._source_rate = None
        self._num_channels = None
        self._in_resampler = None
        self._out_resampler = None
        self._in_buffer = _EMPTY_SAMPLES
        self._pending_denoised_48k = _EMPTY_SAMPLES
        self._out_buffer = _EMPTY_SAMPLES

    def _ensure_pipeline(self, sample_rate: int, num_channels: int) -> None:
        if self._source_rate == sample_rate and self._num_channels == num_channels:
            return

        self._source_rate = sample_rate
        self._num_channels = num_channels
        self._in_buffer = _EMPTY_SAMPLES
        self._pending_denoised_48k = _EMPTY_SAMPLES
        self._out_buffer = _EMPTY_SAMPLES

        if sample_rate == _RNNOISE_SAMPLE_RATE:
            self._in_resampler = None
            self._out_resampler = None
        else:
            self._in_resampler = rtc.AudioResampler(
                sample_rate, _RNNOISE_SAMPLE_RATE, num_channels=num_channels
            )
            self._out_resampler = rtc.AudioResampler(
                _RNNOISE_SAMPLE_RATE, sample_rate, num_channels=num_channels
            )

    def _resample_to_rnnoise(self, frame: rtc.AudioFrame) -> np.ndarray:
        if frame.sample_rate == _RNNOISE_SAMPLE_RATE:
            return np.frombuffer(frame.data, dtype=np.int16)

        assert self._in_resampler is not None
        resampled_frames = self._in_resampler.push(frame)
        if not resampled_frames:
            return _EMPTY_SAMPLES
        return np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in resampled_frames])

    def _denoise_complete_chunks(self) -> None:
        num_chunks = len(self._in_buffer) // _RNNOISE_FRAME_SAMPLES
        if num_chunks == 0:
            return

        drain_len = num_chunks * _RNNOISE_FRAME_SAMPLES
        to_denoise = self._in_buffer[:drain_len]
        self._in_buffer = self._in_buffer[drain_len:]

        denoised_parts = [
            np.asarray(denoised, dtype=np.int16).reshape(-1)
            for _, denoised in self._denoiser.denoise_chunk(to_denoise)
        ]
        if denoised_parts:
            self._pending_denoised_48k = np.concatenate(
                [self._pending_denoised_48k, *denoised_parts]
            )

    def _resample_denoised_to_source(self, source_rate: int) -> None:
        if len(self._pending_denoised_48k) == 0:
            return

        if source_rate == _RNNOISE_SAMPLE_RATE:
            self._out_buffer = np.concatenate([self._out_buffer, self._pending_denoised_48k])
            self._pending_denoised_48k = _EMPTY_SAMPLES
            return

        assert self._out_resampler is not None
        assert self._num_channels is not None
        denoised_frame = rtc.AudioFrame(
            data=self._pending_denoised_48k.tobytes(),
            sample_rate=_RNNOISE_SAMPLE_RATE,
            num_channels=self._num_channels,
            samples_per_channel=len(self._pending_denoised_48k),
        )
        self._pending_denoised_48k = _EMPTY_SAMPLES

        resampled_frames = self._out_resampler.push(denoised_frame)
        if resampled_frames:
            out_samples = np.concatenate(
                [np.frombuffer(f.data, dtype=np.int16) for f in resampled_frames]
            )
            self._out_buffer = np.concatenate([self._out_buffer, out_samples])

    def _take_output_frame(self, frame: rtc.AudioFrame, n: int) -> rtc.AudioFrame:
        if len(self._out_buffer) >= n:
            out_samples = self._out_buffer[:n]
            self._out_buffer = self._out_buffer[n:]
        else:
            # Warm-up latency: the resampler/denoiser hasn't produced enough
            # samples yet. Pad with leading silence and keep the same length.
            pad = np.zeros(n - len(self._out_buffer), dtype=np.int16)
            out_samples = np.concatenate([pad, self._out_buffer])
            self._out_buffer = _EMPTY_SAMPLES

        return rtc.AudioFrame(
            data=out_samples.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=n,
        )


def RNNoise(*, enabled: bool = True) -> RNNoiseFrameProcessor:
    """Create an RNNoise noise-cancellation FrameProcessor."""
    return RNNoiseFrameProcessor(enabled=enabled)
