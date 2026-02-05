# Copyright 2025 LiveKit, Inc.
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

"""Vosk plugin utilities."""

from __future__ import annotations

from livekit import rtc
from livekit.agents.utils import AudioBuffer


def _ensure_frames(buffer: AudioBuffer) -> list[rtc.AudioFrame]:
    if isinstance(buffer, rtc.AudioFrame):
        return [buffer]
    return list(buffer)


def resample_audio(buffer: AudioBuffer, target_sample_rate: int) -> list[rtc.AudioFrame]:
    """Resample audio frames to the target sample rate."""
    frames = _ensure_frames(buffer)
    if not frames:
        return []

    sample_rate = frames[0].sample_rate
    for frame in frames[1:]:
        if frame.sample_rate != sample_rate:
            raise ValueError("audio frames must have a consistent sample rate")

    if sample_rate == target_sample_rate:
        return frames

    resampler = rtc.AudioResampler(
        sample_rate,
        target_sample_rate,
        quality=rtc.AudioResamplerQuality.HIGH,
    )

    out: list[rtc.AudioFrame] = []
    for frame in frames:
        out.extend(resampler.push(frame))

    out.extend(resampler.flush())
    return out
