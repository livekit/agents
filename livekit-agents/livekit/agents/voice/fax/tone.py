from __future__ import annotations

import numpy as np

# ITU-T T.30: a calling fax terminal sends CNG, an 1100 Hz tone that is on for 0.5 s
# and off for 3 s, repeated until a fax terminal answers or the caller gives up.
CNG_FREQUENCY = 1100.0
CNG_PERIOD = 3.5

BLOCK_DURATION = 0.05

# fraction of a block's energy that must sit at the CNG frequency
TONE_ENERGY_RATIO = 0.6
# int16 RMS below which a block is treated as silence
MIN_RMS = 100.0

# 0.5 s nominal, with slack for jitter and block edges
BEEP_MIN_DURATION = 0.35
BEEP_MAX_DURATION = 0.75
# 3.5 s nominal start-to-start spacing
BEEP_MIN_SPACING = 3.0
BEEP_MAX_SPACING = 4.0
BEEPS_REQUIRED = 2


class _CNGToneDetector:
    """Streaming detector for the fax CNG cadence in mono int16 PCM.

    A single 1100 Hz beep is not enough: it requires ``BEEPS_REQUIRED`` beeps of the
    right length at the right spacing, so steady tones, busy signals and speech are
    ignored.
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._block_len = int(sample_rate * BLOCK_DURATION)
        n = np.arange(self._block_len)
        self._kernel = np.exp(-2j * np.pi * CNG_FREQUENCY * n / sample_rate)

        self._buf = np.zeros(0, dtype=np.float64)
        self._elapsed = 0.0
        self._tone_started_at: float | None = None
        self._beep_starts: list[float] = []
        self._detected_at: float | None = None

    @property
    def detected_at(self) -> float | None:
        """Audio time in seconds at which the cadence was confirmed, if it was."""
        return self._detected_at

    def push(self, samples: np.ndarray) -> bool:
        """Feed mono int16 samples. Returns ``True`` once a fax has been detected."""
        if self._detected_at is not None:
            return True

        self._buf = np.concatenate([self._buf, samples.astype(np.float64)])
        while len(self._buf) >= self._block_len:
            block = self._buf[: self._block_len]
            self._buf = self._buf[self._block_len :]
            self._on_block(block)
            self._elapsed += BLOCK_DURATION
            if self._detected_at is not None:
                return True

        return False

    def _is_tone(self, block: np.ndarray) -> bool:
        energy = float(np.dot(block, block))
        if energy <= 0.0 or np.sqrt(energy / len(block)) < MIN_RMS:
            return False

        # for a pure sine at CNG_FREQUENCY, |X|^2 * 2 / N equals the block energy
        tone_energy = float(abs(np.dot(block, self._kernel))) ** 2 * 2.0 / len(block)
        return tone_energy / energy >= TONE_ENERGY_RATIO

    def _on_block(self, block: np.ndarray) -> None:
        if self._is_tone(block):
            if self._tone_started_at is None:
                self._tone_started_at = self._elapsed
            return

        if self._tone_started_at is None:
            return

        started_at = self._tone_started_at
        self._tone_started_at = None
        if not (BEEP_MIN_DURATION <= self._elapsed - started_at <= BEEP_MAX_DURATION):
            return

        if self._beep_starts and not (
            BEEP_MIN_SPACING <= started_at - self._beep_starts[-1] <= BEEP_MAX_SPACING
        ):
            self._beep_starts = []

        self._beep_starts.append(started_at)
        if len(self._beep_starts) >= BEEPS_REQUIRED:
            self._detected_at = self._elapsed
