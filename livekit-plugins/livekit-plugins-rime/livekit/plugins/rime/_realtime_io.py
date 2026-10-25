# SPDX-License-Identifier: Apache-2.0
"""Playback gain for Prism, without changes to shared LiveKit output classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from livekit import rtc
from livekit.agents.voice import io


@dataclass
class AudioOutputCapabilities(io.AudioOutputCapabilities):
    gain: bool = True


class AudioOutput(io.AudioOutput):
    """An output-chain wrapper that changes gain without changing playback timing.

    A downstream Rime gain wrapper owns gain when present. This lets an
    application update gain or replace an output without scaling frames twice.
    """

    def __init__(self, output: io.AudioOutput, *, gain: bool = True) -> None:
        super().__init__(
            label="RimePrismOutput",
            next_in_chain=output,
            capabilities=AudioOutputCapabilities(pause=output.can_pause, gain=gain),
        )
        self._gain_enabled = gain
        self._playback_gain = 1.0
        self._playback_gain_revision = object()

    def _downstream_gain(self) -> AudioOutput | None:
        output = self.next_in_chain
        while output is not None:
            if isinstance(output, AudioOutput):
                return output
            output = output.next_in_chain
        return None

    @property
    def gain(self) -> float:
        output = self._downstream_gain()
        return output.gain if output is not None else self._playback_gain

    @property
    def _gain_revision(self) -> object:
        output = self._downstream_gain()
        return output._gain_revision if output is not None else self._playback_gain_revision

    def set_gain(self, gain: float) -> bool:
        if not math.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError("gain must be finite and between 0 and 1")
        if not self._gain_enabled:
            return False
        output = self._downstream_gain()
        if output is not None:
            return output.set_gain(gain)
        self._playback_gain = gain
        self._playback_gain_revision = object()
        return True

    def _apply_gain(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if self._playback_gain == 1 or self._downstream_gain() is not None:
            return frame
        samples = np.frombuffer(frame.data, dtype=np.int16)
        return rtc.AudioFrame(
            data=(samples * self._playback_gain).astype(np.int16).tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        assert self.next_in_chain is not None
        await self.next_in_chain.capture_frame(self._apply_gain(frame))

    def flush(self) -> None:
        super().flush()
        assert self.next_in_chain is not None
        self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        assert self.next_in_chain is not None
        self.next_in_chain.clear_buffer()
