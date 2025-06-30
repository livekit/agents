from __future__ import annotations

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

from .._exceptions import APIError
from ..log import logger
from ..metrics import TTSMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio, audio, codecs, log_exceptions

lk_dump_tts = int(os.getenv("LK_DUMP_TTS", 0))


@dataclass
class SynthesizedAudio:
    frame: rtc.AudioFrame
    """Synthesized audio frame"""
    request_id: str
    """Request ID (one segment could be made up of multiple requests)"""
    is_final: bool = False
    """Whether this is latest frame of the segment"""
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
    error: Exception = Field(..., exclude=True)
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
    ) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._label = f"{type(self).__module__}.{type(self).__name__}"

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
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream: ...

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
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

        self._tee = aio.itertools.tee(self._event_ch, 2)
        self._event_aiter, monitor_aiter = self._tee
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
    async def _run(self, output_emitter: AudioEmitter) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            output_emitter = AudioEmitter(label=self._tts.label, dst_ch=self._event_ch)
            try:
                await self._run(output_emitter)

                output_emitter.end_input()
                # wait for all audio frames to be pushed & propagate errors
                await output_emitter.join()

                if output_emitter.pushed_duration() <= 0.0:
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
                        extra={"tts": self._tts._label, "attempt": i + 1, "streamed": False},
                    )

                await asyncio.sleep(retry_interval)
                # Reset the flag when retrying
                self._current_attempt_has_error = False
            finally:
                await output_emitter.aclose()

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
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
        await self._tee.aclose()

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

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self._input_ch = aio.Chan[Union[str, SynthesizeStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._tee = aio.itertools.tee(self._event_ch, 2)
        self._event_aiter, self._monitor_aiter = self._tee

        self._task = asyncio.create_task(self._main_task(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._metrics_task: asyncio.Task[None] | None = None  # started on first push
        self._current_attempt_has_error = False
        self._started_time: float = 0
        self._pushed_text: str = ""

        # used to track metrics
        self._mtc_pending_texts: list[str] = []
        self._mtc_text = ""
        self._num_segments = 0

    @abstractmethod
    async def _run(self, output_emitter: AudioEmitter) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            output_emitter = AudioEmitter(label=self._tts.label, dst_ch=self._event_ch)
            try:
                await self._run(output_emitter)

                output_emitter.end_input()
                # wait for all audio frames to be pushed & propagate errors
                await output_emitter.join()

                if self._pushed_text.strip():
                    if output_emitter.pushed_duration(idx=-1) <= 0.0:
                        raise APIError(f"no audio frames were pushed for text: {self._pushed_text}")

                    if self._num_segments != output_emitter.num_segments:
                        raise APIError(
                            f"number of segments mismatch: expected {self._num_segments}, "
                            f"but got {output_emitter.num_segments}"
                        )

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
                        extra={"tts": self._tts._label, "attempt": i + 1, "streamed": True},
                    )

                await asyncio.sleep(retry_interval)
                # Reset the flag when retrying
                self._current_attempt_has_error = False
            finally:
                await output_emitter.aclose()

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
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
        segment_id = ""

        def _emit_metrics() -> None:
            nonlocal audio_duration, ttfb, request_id, segment_id

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
                segment_id=segment_id,
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
            segment_id = ev.segment_id

            if ev.is_final:
                _emit_metrics()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if not token or self._input_ch.closed:
            return

        self._pushed_text += token

        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(
                self._metrics_monitor_task(self._monitor_aiter), name="TTS._metrics_task"
            )

        if not self._mtc_text:
            if self._num_segments >= 1:
                logger.warning(
                    "SynthesizeStream: handling multiple segments in a single instance is "
                    "deprecated. Please create a new SynthesizeStream instance for each segment. "
                    "Most TTS plugins now use pooled WebSocket connections via ConnectionPool."
                )
                return

            self._num_segments += 1

        self._mtc_text += token
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self._input_ch.closed:
            return

        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
        self._input_ch.close()

        if self._metrics_task is not None:
            await self._metrics_task

        await self._tee.aclose()

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


class AudioEmitter:
    class _FlushSegment:
        pass

    @dataclass
    class _StartSegment:
        segment_id: str

    class _EndSegment:
        pass

    @dataclass
    class _SegmentContext:
        segment_id: str
        audio_duration: float = 0.0

    def __init__(
        self,
        *,
        label: str,
        dst_ch: aio.Chan[SynthesizedAudio],
    ) -> None:
        self._dst_ch = dst_ch
        self._label = label
        self._request_id: str = ""
        self._started = False
        self._num_segments = 0
        self._audio_durations: list[float] = []  # track durations per segment

    def pushed_duration(self, idx: int = -1) -> float:
        return (
            self._audio_durations[idx]
            if -len(self._audio_durations) <= idx < len(self._audio_durations)
            else 0.0
        )

    @property
    def num_segments(self) -> int:
        return self._num_segments

    def initialize(
        self,
        *,
        request_id: str,
        sample_rate: int,
        num_channels: int,
        mime_type: str,
        frame_size_ms: int = 200,
        stream: bool = False,
    ) -> None:
        if self._started:
            raise RuntimeError("AudioEmitter already started")

        self._is_raw_pcm = False
        if mime_type:
            mt = mime_type.lower().strip()
            self._is_raw_pcm = mt.startswith("audio/pcm") or mt.startswith("audio/raw")

        self._mime_type = mime_type

        if not request_id:
            logger.warning("no request_id provided for TTS %s", self._label)
            request_id = "unknown"

        self._started = True
        self._request_id = request_id
        self._frame_size_ms = frame_size_ms
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._streaming = stream

        self._write_ch = aio.Chan[
            Union[
                bytes,
                AudioEmitter._FlushSegment,
                AudioEmitter._StartSegment,
                AudioEmitter._EndSegment,
            ]
        ]()
        self._main_atask = asyncio.create_task(self._main_task(), name="AudioEmitter._main_task")

        if not self._streaming:
            self.__start_segment(segment_id="")  # always start a segment with stream=False

    def start_segment(self, *, segment_id: str) -> None:
        if not self._streaming:
            raise RuntimeError(
                "start_segment() can only be called when SynthesizeStream is initialized "
                "with stream=True"
            )

        return self.__start_segment(segment_id=segment_id)

    def __start_segment(self, *, segment_id: str) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self._num_segments += 1
        self._write_ch.send_nowait(self._StartSegment(segment_id=segment_id))

    def end_segment(self) -> None:
        if not self._streaming:
            raise RuntimeError(
                "end_segment() can only be called when SynthesizeStream is initialized "
                "with stream=True"
            )

        return self.__end_segment()

    def __end_segment(self) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self._write_ch.send_nowait(self._EndSegment())

    def push(self, data: bytes) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self._write_ch.send_nowait(data)

    def flush(self) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self._write_ch.send_nowait(self._FlushSegment())

    def end_input(self) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self.__end_segment()
        self._write_ch.close()

    async def join(self) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        await self._main_atask

    async def aclose(self) -> None:
        if not self._started:
            return

        await aio.cancel_and_wait(self._main_atask)

    @log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        audio_decoder: codecs.AudioStreamDecoder | None = None
        decode_atask: asyncio.Task | None = None
        segment_ctx: AudioEmitter._SegmentContext | None = None
        last_frame: rtc.AudioFrame | None = None
        debug_frames: list[rtc.AudioFrame] = []

        def _emit_frame(frame: rtc.AudioFrame | None = None, *, is_final: bool = False) -> None:
            nonlocal last_frame, segment_ctx
            assert segment_ctx is not None

            if last_frame is None:
                if not is_final:
                    last_frame = frame
                    return
                elif segment_ctx.audio_duration > 0:
                    if frame is None:
                        # NOTE: if end_input called after flush with no new audio frames pushed,
                        # it will create a 0.01s empty frame to indicate the end of the segment
                        frame = rtc.AudioFrame(
                            data=b"\0\0" * (self._sample_rate // 100 * self._num_channels),
                            sample_rate=self._sample_rate,
                            num_channels=self._num_channels,
                            samples_per_channel=self._sample_rate // 100,
                        )
                    else:
                        segment_ctx.audio_duration += frame.duration
                        self._audio_durations[-1] += frame.duration

                        if lk_dump_tts:
                            debug_frames.append(frame)

                    self._dst_ch.send_nowait(
                        SynthesizedAudio(
                            frame=frame,
                            request_id=self._request_id,
                            segment_id=segment_ctx.segment_id,
                            is_final=True,
                        )
                    )
                    return

            if last_frame is not None:
                self._dst_ch.send_nowait(
                    SynthesizedAudio(
                        frame=last_frame,
                        request_id=self._request_id,
                        segment_id=segment_ctx.segment_id,
                        is_final=is_final,
                    )
                )
                segment_ctx.audio_duration += last_frame.duration
                self._audio_durations[-1] += last_frame.duration

                if lk_dump_tts:
                    debug_frames.append(last_frame)

            last_frame = frame

        def _flush_frame() -> None:
            nonlocal last_frame, segment_ctx
            assert segment_ctx is not None

            if last_frame is None:
                return

            self._dst_ch.send_nowait(
                SynthesizedAudio(
                    frame=last_frame,
                    request_id=self._request_id,
                    segment_id=segment_ctx.segment_id,
                    is_final=False,  # flush isn't final
                )
            )
            segment_ctx.audio_duration += last_frame.duration
            self._audio_durations[-1] += last_frame.duration

            if lk_dump_tts:
                debug_frames.append(last_frame)

            last_frame = None

        def dump_segment() -> None:
            nonlocal segment_ctx
            assert segment_ctx is not None

            if not lk_dump_tts or not debug_frames:
                return

            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = (
                f"lk_dump/{self._label}_{self._request_id}_{segment_ctx.segment_id}_{ts}.wav"
                if self._streaming
                else f"lk_dump/{self._label}_{self._request_id}_{ts}.wav"
            )
            with open(fname, "wb") as f:
                f.write(rtc.combine_audio_frames(debug_frames).to_wav_bytes())

            debug_frames.clear()

        @log_exceptions(logger=logger)
        async def _decode_task() -> None:
            nonlocal audio_decoder, segment_ctx
            assert segment_ctx is not None
            assert audio_decoder is not None

            audio_byte_stream: audio.AudioByteStream | None = None
            async for frame in audio_decoder:
                if audio_byte_stream is None:
                    audio_byte_stream = audio.AudioByteStream(
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=int(frame.sample_rate // 1000 * self._frame_size_ms),
                    )
                for f in audio_byte_stream.push(frame.data):
                    _emit_frame(f)

            if audio_byte_stream:
                for f in audio_byte_stream.flush():
                    _emit_frame(f)

            await audio_decoder.aclose()

        audio_byte_stream: audio.AudioByteStream | None = None
        try:
            async for data in self._write_ch:
                if isinstance(data, AudioEmitter._StartSegment):
                    if segment_ctx:
                        raise RuntimeError(
                            "start_segment() called before the previous segment was ended"
                        )

                    self._audio_durations.append(0.0)
                    segment_ctx = AudioEmitter._SegmentContext(segment_id=data.segment_id)
                    continue

                if not segment_ctx:
                    if self._streaming:
                        if isinstance(data, (AudioEmitter._EndSegment, AudioEmitter._FlushSegment)):
                            continue  # empty segment, ignore

                        raise RuntimeError(
                            "start_segment() must be called before pushing audio data"
                        )

                if self._is_raw_pcm:
                    if isinstance(data, bytes):
                        if audio_byte_stream is None:
                            audio_byte_stream = audio.AudioByteStream(
                                sample_rate=self._sample_rate,
                                num_channels=self._num_channels,
                                samples_per_channel=int(
                                    self._sample_rate // 1000 * self._frame_size_ms
                                ),
                            )

                        for f in audio_byte_stream.push(data):
                            _emit_frame(f)
                    elif audio_byte_stream:
                        if isinstance(data, AudioEmitter._FlushSegment):
                            for f in audio_byte_stream.flush():
                                _emit_frame(f)
                            _flush_frame()

                        elif isinstance(data, AudioEmitter._EndSegment):
                            for f in audio_byte_stream.flush():
                                _emit_frame(f)

                            _emit_frame(is_final=True)
                            dump_segment()
                            segment_ctx = audio_byte_stream = last_frame = None
                        else:
                            logger.warning("unknown data type: %s", type(data))
                else:
                    if isinstance(data, bytes):
                        if not audio_decoder:
                            audio_decoder = codecs.AudioStreamDecoder(
                                sample_rate=self._sample_rate,
                                num_channels=self._num_channels,
                                format=self._mime_type,
                            )
                            decode_atask = asyncio.create_task(_decode_task())
                        audio_decoder.push(data)
                    elif decode_atask:
                        if isinstance(data, AudioEmitter._FlushSegment) and audio_decoder:
                            audio_decoder.end_input()
                            await decode_atask
                            _flush_frame()
                            audio_decoder = None

                        elif isinstance(data, AudioEmitter._EndSegment) and segment_ctx:
                            if audio_decoder:
                                audio_decoder.end_input()
                                await decode_atask
                            _emit_frame(is_final=True)
                            dump_segment()
                            audio_decoder = segment_ctx = audio_byte_stream = last_frame = None
                        else:
                            logger.warning("unknown data type: %s", type(data))

        finally:
            if audio_decoder and decode_atask:
                await audio_decoder.aclose()
                await aio.cancel_and_wait(decode_atask)
