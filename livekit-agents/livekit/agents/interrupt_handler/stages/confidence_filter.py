from collections import deque
import os

class ConfidenceFilter:
    """
    Enhanced confidence filtering:
    - Rolling average smoothing
    - Minimum spike threshold
    - ENV-configurable threshold
    - Ignores extremely low ASR noise
    """

    def __init__(self, min_confidence: float = None, window: int = None):
        # Allow ENV override
        env_threshold = os.getenv("CONFIDENCE_THRESHOLD")
        env_window = os.getenv("CONFIDENCE_WINDOW")

        self.min_confidence = (
            float(env_threshold) if env_threshold else (min_confidence or 0.6)
        )

        self.window = int(env_window) if env_window else (window or 3)

        # Rolling window
        self.history = deque(maxlen=self.window)

        # Safety threshold — ignore pure noise like 0.01
        self.min_absolute_cutoff = 0.15

    def is_confident(self, transcript: str, confidence: float) -> bool:
        """
        Returns True if:
            1. Raw confidence is not extreme noise (cutoff)
            2. Rolling smoothed confidence >= threshold
        """

        # Absolute cutoff → noise / breathing / mic pops
        if confidence < self.min_absolute_cutoff:
            return False

        # Add sample to smoothing window
        self.history.append(confidence)

        # Rolling mean
        smoothed = sum(self.history) / len(self.history)

        return smoothed >= self.min_confidence