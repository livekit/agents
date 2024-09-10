import numpy as np
from livekit import rtc

# This is the magic number during testing that we use to determine if a frame is loud enough
# to possibly contain speech. It's very conservative.
MAGIC_NUMBER_THRESHOLD = 0.004


class BasicAudioEnergyFilter:
    def __init__(self, *, cooldown_seconds: float = 1):
        self._cooldown_seconds = cooldown_seconds
        self._cooldown = cooldown_seconds

    def push_frame(self, frame: rtc.AudioFrame) -> bool:
        arr = np.frombuffer(frame.data, dtype=np.int16)
        float_arr = arr.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(float_arr)))
        if rms > MAGIC_NUMBER_THRESHOLD:
            self._cooldown = self._cooldown_seconds
            return True

        duration_seconds = frame.samples_per_channel / frame.sample_rate
        self._cooldown -= duration_seconds
        if self._cooldown > 0:
            return True

        return False
