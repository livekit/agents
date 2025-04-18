import asyncio
from abc import ABC, abstractmethod


class FrameSampler(ABC):
    @abstractmethod
    def allow(self, frame) -> bool:
        """returns True if frame should be sent."""
        ...


class FpsSampler(FrameSampler):
    def __init__(self, fps: float):
        if fps <= 0:
            self._interval = float("inf")
        else:
            self._interval = 1.0 / fps
        self._last_time = 0.0

    def allow(self, frame) -> bool:
        now = asyncio.get_event_loop().time()
        if now - self._last_time < self._interval:
            return False
        self._last_time = now
        return True
