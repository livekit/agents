from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, Protocol

from livekit import rtc


@dataclass
class TextChunk:
    text: str
    is_final: bool


AudioStream = AsyncIterable[rtc.AudioFrame]
VideoStream = AsyncIterable[rtc.VideoFrame]
TextStream = AsyncIterable[TextChunk]


class AudioSink(Protocol):
    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...

    def flush(self) -> None: ...


class TextSink(Protocol):
    async def capture_text(self, text: str) -> None: ...

    def flush(self) -> None: ...


class VideoSink(Protocol):
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    def flush(self) -> None: ...
