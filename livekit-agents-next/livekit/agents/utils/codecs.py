"""Audio stream decoding into PCM frames; the decoder and its buffer are the core's."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from livekit import rtc

from .. import _core
from ..log import logger
from . import aio

__all__ = [
    "AudioStreamDecoder",
    "StreamBuffer",
]


StreamBuffer = _core.StreamBuffer


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
        self._decoder = _core.AudioDecoder(
            format.lower() if format else "",
            sample_rate=sample_rate,
            num_channels=num_channels or 1,
        )
        self._output_ch = aio.Chan[rtc.AudioFrame]()
        loop = asyncio.get_event_loop()
        self._decoder.set_on_output(lambda: loop.call_soon_threadsafe(self._drain))

    def _drain(self) -> None:
        self.push(b"")

    def _emit(self, pcm: bytes) -> None:
        if pcm and (fmt := self._decoder.output_format) is not None:
            sample_rate, num_channels = fmt
            self._output_ch.send_nowait(
                rtc.AudioFrame(
                    data=pcm,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=len(pcm) // (2 * num_channels),
                )
            )

    def push(self, chunk: bytes) -> None:
        if self._output_ch.closed:
            return
        try:
            self._emit(self._decoder.push(chunk))
        except RuntimeError:
            logger.exception("error decoding audio")
            self._output_ch.close()

    def end_input(self) -> None:
        if self._output_ch.closed:
            return
        try:
            self._emit(self._decoder.flush())
        except RuntimeError:
            logger.exception("error decoding audio")
        self._decoder.set_on_output(None)
        self._output_ch.close()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._output_ch.__anext__()

    async def aclose(self) -> None:
        self._decoder.set_on_output(None)
        self._output_ch.close()
        await self._decoder.aclose()
