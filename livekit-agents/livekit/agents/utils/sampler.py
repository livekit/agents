from typing import Protocol

from livekit import rtc


class FrameSampler(Protocol):
    def allow(self, frame: rtc.VideoFrame) -> bool:
        """Return True if this frame should be sent."""
