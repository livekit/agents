from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from livekit import rtc

# a floor this low is digital silence; it keeps the gate's ratios finite when a source emits zeros
_SILENCE_FLOOR = 1e-4


def frame_rms(frame: rtc.AudioFrame) -> float:
    """RMS of a 16-bit frame, normalized to [0, 1]."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(samples)))) / 32768.0 if samples.size else 0.0


class AudioGate(Protocol):
    """Decides which frames of an audio source carry speech.

    Used on a model's output (duplex adapter, translation) and on participants' microphones
    (transcription session).
    """

    def update(self, frame: rtc.AudioFrame) -> bool:
        """True while the frame belongs to an open burst of speech."""
        ...

    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the source."""
        ...


class FixedGate:
    """Opens on output that stands out from a silence the plugin already knows."""

    def __init__(
        self,
        silence: float,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
    ) -> None:
        self._floor = max(silence, _SILENCE_FLOOR)
        self._activation_ratio = activation_ratio
        self._deactivation_ratio = deactivation_ratio
        self._min_silence_duration = min_silence_duration
        self._open = False
        self._quiet = 0.0

    def deactivate(self) -> None:
        self._open = False
        self._quiet = 0.0

    def update(self, frame: rtc.AudioFrame) -> bool:
        rms = frame_rms(frame)

        if not self._open:
            if rms > self._floor * self._activation_ratio:
                self._open = True
                self._quiet = 0.0
        elif rms < self._floor * self._deactivation_ratio:
            self._quiet += frame.duration
            if self._quiet >= self._min_silence_duration:
                self._open = False
        else:
            self._quiet = 0.0

        return self._open


class AdaptiveNoiseGate:
    """Opens on output that stands out from the model's own silence.

    The floor is the quietest ``min_silence_duration`` stretch the model produced while it was
    not speaking, within ``window``: speech never raises it, and no single frame defines it.
    Thresholds are ratios against it and durations count audio rather than wall clock, so one
    set of defaults ports across providers, frame sizes and networks.

    ``max_floor`` caps the floor, for a source that may start with speech rather than silence
    (a microphone joining mid-sentence): without it, that speech is learned as the floor and
    the gate stays closed until a quieter stretch comes along.
    """

    def __init__(
        self,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
        window: float = 10.0,
        max_floor: float | None = None,
    ) -> None:
        self._activation_ratio = activation_ratio
        self._max_floor = max_floor
        self._deactivation_ratio = deactivation_ratio
        self._min_silence_duration = min_silence_duration
        self._window = window
        self._history: deque[tuple[float, float]] = deque()
        self._history_duration = 0.0
        self._stretch_sum = 0.0
        self._stretch_duration = 0.0
        self._open = False
        self._quiet = 0.0

    def deactivate(self) -> None:
        self._open = False
        self._quiet = 0.0

    def update(self, frame: rtc.AudioFrame) -> bool:
        rms = frame_rms(frame)

        if not self._open:
            # only silence teaches the floor, and only a whole stretch of it
            self._stretch_sum += rms * frame.duration
            self._stretch_duration += frame.duration
            if self._stretch_duration >= self._min_silence_duration:
                mean = self._stretch_sum / self._stretch_duration
                self._history.append((mean, self._stretch_duration))
                self._history_duration += self._stretch_duration
                self._stretch_sum = self._stretch_duration = 0.0
                while self._history_duration > self._window and len(self._history) > 1:
                    self._history_duration -= self._history.popleft()[1]

        if self._history:
            floor = min(level for level, _ in self._history)
        elif self._stretch_duration:
            floor = self._stretch_sum / self._stretch_duration
        else:
            floor = rms
        if self._max_floor is not None:
            floor = min(floor, self._max_floor)
        floor = max(floor, _SILENCE_FLOOR)

        if not self._open:
            if rms > floor * self._activation_ratio:
                self._open = True
                self._quiet = 0.0
        elif rms < floor * self._deactivation_ratio:
            self._quiet += frame.duration
            if self._quiet >= self._min_silence_duration:
                self._open = False
        else:
            self._quiet = 0.0

        return self._open
