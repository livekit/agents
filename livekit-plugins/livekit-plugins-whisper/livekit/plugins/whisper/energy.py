from enum import Enum

import numpy as np
from livekit import rtc

MAGIC_NUMBER_THRESHOLD = 0.004**2


class AudioEnergyFilter:
    class State(Enum):
        START = 0
        SPEAKING = 1
        SILENCE = 2
        END = 3

    def __init__(
        self, *, min_silence: float = 1.5, rms_threshold: float = MAGIC_NUMBER_THRESHOLD
    ):
        self._cooldown_seconds = min_silence
        self._cooldown = min_silence
        self._state = self.State.SILENCE
        self._rms_threshold = rms_threshold

    def update(self, frame: rtc.AudioFrame) -> State:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.mean(np.square(float_arr))

        if rms > self._rms_threshold:
            self._cooldown = self._cooldown_seconds
            if self._state in (self.State.SILENCE, self.State.END):
                self._state = self.State.START
            else:
                self._state = self.State.SPEAKING
        else:
            if self._cooldown <= 0:
                if self._state in (self.State.SPEAKING, self.State.START):
                    self._state = self.State.END
                elif self._state == self.State.END:
                    self._state = self.State.SILENCE
            else:
                # keep speaking during cooldown
                self._cooldown -= frame.duration
                self._state = self.State.SPEAKING

        return self._state
