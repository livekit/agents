from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Literal, Union

from livekit import rtc

from ..metrics import TTSMetrics
from ..utils import aio, audio


@dataclass
class SynthesizedAudio:
    frame: rtc.AudioFrame
    """Synthesized audio frame"""
    request_id: str
    """Request ID (one segment could be made up of multiple requests)"""
    is_final: bool = False
    """Whether this is latest frame of the segment (streaming only)"""
    segment_id: str = ""
    """Segment ID, each segment is separated by a flush (streaming only)"""
    delta_text: str = ""
    """Current segment of the synthesized audio (streaming only)"""


@dataclass
class TTSCapabilities:
    streaming: bool


class TTS(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(
        self, *, capabilities: TTSCapabilities, sample_rate: int, num_channels: int
    ) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def capabilities(self) -> TTSCapabilities:
        return self._capabilities

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @abstractmethod
    def synthesize(self, text: str) -> ChunkedStream: ...

    def stream(self) -> SynthesizeStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    async def aclose(self) -> None: ...


class ChunkedStream(ABC):
    """Used by the non-streamed synthesize API, some providers support chunked http responses"""

    def __init__(self, tts: TTS, text: str) -> None:
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._tts = tts
        self._input_text = text

        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="TTS._metrics_task"
        )

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        """Task used to collect metrics"""

        start_time = time.perf_counter()
        audio_duration = 0.0
        ttfb = -1.0
        request_id = ""

        async for ev in event_aiter:
            request_id = ev.request_id
            if ttfb == -1.0:
                ttfb = time.perf_counter() - start_time

            audio_duration += ev.frame.duration

        duration = time.perf_counter() - start_time
        metrics = TTSMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttfb=ttfb,
            duration=duration,
            characters_count=len(self._input_text),
            audio_duration=audio_duration,
            cancelled=self._task.cancelled(),
            label=self._tts._label,
            streamed=False,
            error=None,
        )
        self._tts.emit("metrics_collected", metrics)

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        frames = []
        async for ev in self:
            frames.append(ev.frame)
        return audio.merge_frames(frames)

    @abstractmethod
    async def _main_task(self) -> None: ...

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.gracefully_cancel(self._task)
        self._event_ch.close()
        await self._metrics_task

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if self._task.done() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self


class SynthesizeStream(ABC):
    class _FlushSentinel: ...

    def __init__(self, tts: TTS) -> None:
        self._tts = tts
        self._input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._event_aiter, self._monitor_aiter = aio.itertools.tee(self._event_ch, 2)

        self._task = asyncio.create_task(self._main_task(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._metrics_task: asyncio.Task | None = None  # started on first push

        # used to track metrics
        self._mtc_pending_texts: list[str] = []
        self._mtc_text = ""

    @abstractmethod
    async def _main_task(self) -> None: ...

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SynthesizedAudio]
    ) -> None:
        """Task used to collect metrics"""
        start_time = time.perf_counter()
        audio_duration = 0.0
        ttfb = -1.0
        request_id = ""

        def _emit_metrics():
            nonlocal start_time, audio_duration, ttfb, request_id
            duration = time.perf_counter() - start_time

            if not self._mtc_pending_texts:
                return

            text = self._mtc_pending_texts.pop(0)
            if not text:
                return

            metrics = TTSMetrics(
                timestamp=time.time(),
                request_id=request_id,
                ttfb=ttfb,
                duration=duration,
                characters_count=len(text),
                audio_duration=audio_duration,
                cancelled=self._task.cancelled(),
                label=self._tts._label,
                streamed=True,
                error=None,
            )
            self._tts.emit("metrics_collected", metrics)

            audio_duration = 0.0
            ttfb = -1.0
            request_id = ""
            start_time = time.perf_counter()

        async for ev in event_aiter:
            if ttfb == -1.0:
                ttfb = time.perf_counter() - start_time

            audio_duration += ev.frame.duration
            request_id = ev.request_id

            if ev.is_final:
                _emit_metrics()

        if request_id:
            _emit_metrics()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(
                self._metrics_monitor_task(self._monitor_aiter),
                name="TTS._metrics_task",
            )

        self._mtc_text += token
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.gracefully_cancel(self._task)

        if self._metrics_task is not None:
            await self._metrics_task

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if self._task.done() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self
