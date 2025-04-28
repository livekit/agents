from __future__ import annotations

import os
import asyncio
import datetime
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..metrics import TTSMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio, codecs, audio, log_exceptions

lk_dump_tts = int(os.getenv("LK_DUMP_TTS", 0))


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
    """Whether this TTS supports streaming (generally using websockets)"""


class TTSError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["tts_error"] = "tts_error"
    timestamp: float
    label: str
    error: APIError = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class TTS(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]],
    Generic[TEvent],
):
    def __init__(
        self,
        *,
        capabilities: TTSCapabilities,
        sample_rate: int,
        num_channels: int,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS

    @property
    def label(self) -> str:
        return self._label

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
    def synthesize(self, text: str, *, conn_options: APIConnectOptions) -> ChunkedStream: ...

    def stream(self, *, conn_options: APIConnectOptions | None = None) -> SynthesizeStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"  # noqa: E501
        )

    def prewarm(self) -> None:
        """Pre-warm connection to the TTS service"""
        pass

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> TTS:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class ChunkedStream(ABC):
    """Used by the non-streamed synthesize API, some providers support chunked http responses"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        self._event_ch = aio.Chan[SynthesizedAudio]()

        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._current_attempt_has_error = False
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="TTS._metrics_task"
        )
        self._synthesize_task = asyncio.create_task(self._main_task(), name="TTS._synthesize_task")
        self._synthesize_task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def input_text(self) -> str:
        return self._input_text

    @property
    def done(self) -> bool:
        return self._synthesize_task.done()

    @property
    def exception(self) -> BaseException | None:
        return self._synthesize_task.exception()

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
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

        if self._current_attempt_has_error:
            return

        metrics = TTSMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttfb=ttfb,
            duration=duration,
            characters_count=len(self._input_text),
            audio_duration=audio_duration,
            cancelled=self._synthesize_task.cancelled(),
            label=self._tts._label,
            streamed=False,
        )
        self._tts.emit("metrics_collected", metrics)

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        frames = []
        async for ev in self:
            frames.append(ev.frame)

        return rtc.combine_audio_frames(frames)

    @abstractmethod
    async def _run(self, output_emitter: SynthesizedAudioEmitter) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            output_emitter = SynthesizedAudioEmitter(label=self._tts.label, dst_ch=self._event_ch)
            try:
                await self._run(output_emitter)

                # wait for all audio frames to be pushed & propagate errors
                await output_emitter.join()

                if output_emitter.pushed_duration <= 0.0:
                    raise APIError("no audio frames were pushed")

                return
            except APIError as e:
                retry_interval = self._conn_options._interval_for_retry(i)
                if self._conn_options.max_retry == 0 or self._conn_options.max_retry == i:
                    self._emit_error(e, recoverable=False)
                    raise
                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to synthesize speech, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._tts._label,
                            "attempt": i + 1,
                            "streamed": False,
                        },
                    )

                await asyncio.sleep(retry_interval)
                # Reset the flag when retrying
                self._current_attempt_has_error = False
            finally:
                await output_emitter.aclose()

    def _emit_error(self, api_error: APIError, recoverable: bool):
        self._current_attempt_has_error = True
        self._tts.emit(
            "error",
            TTSError(
                timestamp=time.time(),
                label=self._tts._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.cancel_and_wait(self._synthesize_task)
        self._event_ch.close()
        await self._metrics_task

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (exc := self._synthesize_task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __aenter__(self) -> ChunkedStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SynthesizeStream(ABC):
    class _FlushSentinel: ...

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions | None = None) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options or DEFAULT_API_CONNECT_OPTIONS
        self._input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._event_aiter, self._monitor_aiter = aio.itertools.tee(self._event_ch, 2)

        self._task = asyncio.create_task(self._main_task(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._metrics_task: asyncio.Task | None = None  # started on first push
        self._current_attempt_has_error = False
        self._started_time: float = 0

        # used to track metrics
        self._mtc_pending_texts: list[str] = []
        self._mtc_text = ""

    @abstractmethod
    async def _run(self, output_emitter: SynthesizedAudioEmitter) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                retry_interval = self._conn_options._interval_for_retry(i)
                if self._conn_options.max_retry == 0:
                    self._emit_error(e, recoverable=False)
                    raise
                elif i == self._conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to synthesize speech after {self._conn_options.max_retry + 1} attempts",  # noqa: E501
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to synthesize speech, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "tts": self._tts._label,
                            "attempt": i + 1,
                            "streamed": True,
                        },
                    )

                await asyncio.sleep(retry_interval)
                # Reset the flag when retrying
                self._current_attempt_has_error = False

    def _emit_error(self, api_error: APIError, recoverable: bool):
        self._current_attempt_has_error = True
        self._tts.emit(
            "error",
            TTSError(
                timestamp=time.time(),
                label=self._tts._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    def _mark_started(self) -> None:
        # only set the started time once, it'll get reset after we emit metrics
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        """Task used to collect metrics"""
        audio_duration = 0.0
        ttfb = -1.0
        request_id = ""

        def _emit_metrics():
            nonlocal audio_duration, ttfb, request_id

            if not self._started_time or self._current_attempt_has_error:
                return

            duration = time.perf_counter() - self._started_time

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
            )
            self._tts.emit("metrics_collected", metrics)

            audio_duration = 0.0
            ttfb = -1.0
            request_id = ""
            self._started_time = 0

        async for ev in event_aiter:
            if ttfb == -1.0:
                ttfb = time.perf_counter() - self._started_time

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
        await aio.cancel_and_wait(self._task)

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
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __aenter__(self) -> SynthesizeStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SynthesizedAudioEmitter:
    """
    Utility for buffering and emitting audio frames with metadata to a channel.
    This class helps TTS implementers to correctly handle is_final logic when streaming responses.
    """

    def __init__(
        self,
        *,
        label: str,
        dst_ch: aio.Chan[SynthesizedAudio],
    ) -> None:
        self._dst_ch = dst_ch
        self._label = label
        self._audio_decoder: codecs.AudioStreamDecoder | None = None
        self._audio_byte_stream: audio.AudioByteStream | None = None
        self._frame: rtc.AudioFrame | None = None
        self._request_id: str | None = None
        self._segment_id: str | None = None
        self._started = False
        self._pushed_duration = 0.0

        # when lk_dump_tts is 1
        self._dump_tts_frames = []

    @property
    def pushed_duration(self) -> float:
        return self._pushed_duration

    def start(
        self,
        *,
        request_id: str | None = None,
        segment_id: str | None = None,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        is_raw_pcm: bool = False,
        frame_size_ms: int = 200,
    ) -> None:
        if self._started:
            raise RuntimeError("AudioEmitter already started")

        if is_raw_pcm and (sample_rate is None or num_channels is None):
            raise ValueError("sample_rate and num_channels must be provided if is_raw_pcm is True")

        if not request_id:
            logger.warning("No request_id provided for TTS %s", self._label)
            request_id = "unknown"

        if not segment_id:
            segment_id = ""  # defaults to empty (only available in streaming mode)

        self._request_id = request_id
        self._segment_id = segment_id
        self._started = True
        self._frame_size_ms = frame_size_ms
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if not is_raw_pcm:
            self._audio_decoder = codecs.AudioStreamDecoder(
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

        self._write_ch = aio.Chan[bytes]()
        self._main_atask = asyncio.create_task(
            self._main_task(), name="SynthesizedAudioEmitter._main_task"
        )

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._audio_decoder is not None:
            async for frame in self._audio_decoder:
                if self._audio_byte_stream is None:
                    self._audio_byte_stream = audio.AudioByteStream(
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=int(frame.sample_rate // 1000 * self._frame_size_ms),
                    )

                for f in self._audio_byte_stream.push(frame.data):
                    self._emit_frame(f)

            if self._audio_byte_stream is not None:
                for f in self._audio_byte_stream.flush():
                    self._emit_frame(f)
        else:
            assert self._sample_rate is not None, "sample_rate isn't None if is_raw_pcm is True"
            assert self._num_channels is not None, "num_channels isn't None if is_raw_pcm is True"
            self._audio_byte_stream = audio.AudioByteStream(
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=int(self._sample_rate // 1000 * self._frame_size_ms),
            )

            async for data in self._write_ch:
                for f in self._audio_byte_stream.push(data):
                    self._emit_frame(f)

            for f in self._audio_byte_stream.flush():
                self._emit_frame(f)

        self._emit_frame(is_final=True)

    def push(self, data: bytes) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._audio_decoder is not None:
            self._audio_decoder.push(data)
        else:
            self._write_ch.send_nowait(data)

    def flush(self):
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._audio_decoder is not None:
            self._audio_decoder.end_input()
        else:
            self._write_ch.close()

    async def join(self) -> None:
        if not self._started:
            return

        await self._main_atask  # this will also raise any error that occurred in the main task

    async def aclose(self) -> None:
        if not self._started:
            return

        await aio.cancel_and_wait(self._main_atask)

        if self._audio_decoder is not None:
            await self._audio_decoder.aclose()

        if lk_dump_tts:
            if len(self._dump_tts_frames) == 0:
                logger.warning("No frames to dump for TTS %s", self._label)
                return

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"lk_dump/tts_{self._label}_{self._request_id}_{timestamp}.wav"
            with open(filename, "wb") as f:
                f.write(rtc.combine_audio_frames(self._dump_tts_frames).to_wav_bytes())

    def _emit_frame(self, frame: rtc.AudioFrame | None = None, *, is_final: bool = False):
        if self._frame is None:
            self._frame = frame
            return

        assert self._request_id is not None
        assert self._segment_id is not None

        self._dst_ch.send_nowait(
            SynthesizedAudio(
                frame=self._frame,
                request_id=self._request_id,
                segment_id=self._segment_id,
                is_final=is_final,
            )
        )

        if lk_dump_tts:
            self._dump_tts_frames.append(self._frame)

        self._pushed_duration += self._frame.duration
        self._frame = frame
