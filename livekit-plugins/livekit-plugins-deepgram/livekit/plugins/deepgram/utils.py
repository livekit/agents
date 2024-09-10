import numpy as np
from livekit import rtc


class BasicAudioEnergyFilter:
    def __init__(self, *, threshold: float = 0.1, cooldown_seconds: float = 1):
        self.threshold = threshold
        self._cooldown = 1

    def push_frame(self, frame: rtc.AudioFrame) -> bool:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        if np.sum(arr**2) > self.threshold:
            self._cooldown = 1
            return True

        duration_seconds = frame.samples_per_channel / frame.sample_rate
        self._cooldown -= duration_seconds
        if self._cooldown > 0:
            return True

        return False
