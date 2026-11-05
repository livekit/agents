"""Audio stream decoding into PCM frames; the decoder and its buffer are the core's."""

from __future__ import annotations

from collections.abc import AsyncIterator

from livekit import rtc

from .. import _core
from ..log import logger

__all__ = [
    "AudioStreamDecoder",
]


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream. The decoding is the core's (``_core.AudioDecoder``):
    its frames come out as soon as they are decoded.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = 48000,
        num_channels: int | None = 1,
        format: str | None = None,
    ):
        """
        Args:
            sample_rate: The output rate; the source's when ``None``.
            num_channels: The output channel count.
            format: The input's media type (``"audio/mpeg"``); probed when ``None``.
        """
        self._decoder: _core.AudioDecoder | None = None
        self._ended = False
        try:
            self._decoder = _core.AudioDecoder(
                format.lower() if format else "",
                sample_rate=sample_rate,
                num_channels=num_channels or 1,
            )
        except RuntimeError:
            logger.exception("error decoding audio")

    def push(self, chunk: bytes) -> None:
        if self._decoder is None or self._ended:
            return
        try:
            self._decoder.push(chunk)
        except RuntimeError:
            logger.exception("error decoding audio")
            self.end_input()

    def end_input(self) -> None:
        self._ended = True
        if self._decoder is not None:
            self._decoder.end_input()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame: rtc.AudioFrame | None = None
        if self._decoder is not None:
            try:
                frame = await self._decoder.next()
            except RuntimeError:
                logger.exception("error decoding audio")
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def aclose(self) -> None:
        decoder, self._decoder = self._decoder, None
        if decoder is not None:
            await decoder.aclose()
