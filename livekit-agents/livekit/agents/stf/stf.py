from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Union

from livekit import rtc

from ..utils import aio, misc


@dataclass
class SynthesizedFace:
    request_id: str
    """Request ID (one segment could be made up of multiple requests)"""
    segment_id: str
    """Segment ID, each segment is separated by a flush"""
    frame: rtc.VideoFrame
    """Synthesized video frame"""


class STF(ABC):
    def __init__(
        self, *, width: int, height: int, frame_rate: int
    ) -> None:
        self._width = width
        self._height = height
        self._frame_rate = frame_rate

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @abstractmethod
    def synthesize(self, text: str) -> ChunkedStream: ...

