from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .. import _core
from .._exceptions import APIError, APIStatusError
from ..log import logger
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    APIConnectOptions,
    TimedString,
)
from ..utils import aio, audio, codecs, log_exceptions, shortuuid

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata
    from ..voice.agent_session import SpeechSteeringOptions


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

    @property
    def _core_started_time(self) -> float | None:
        return self.frame.userdata.get(USERDATA_TTS_STARTED_TIME)  # type: ignore[no-any-return]

    @property
    def _core_timed_transcript(self) -> list[TimedString]:
        words = self.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT) or []
        return [w for w in words if isinstance(w, TimedString)]


@dataclass
class TTSCapabilities:
    streaming: bool
    """Whether this TTS supports streaming (generally using websockets)"""
    aligned_transcript: bool = False
    """Whether this TTS supports aligned transcripts with word timestamps"""


@dataclass
class MarkupInfo:
    """What the expressive markup pipeline can do with a given voice."""

    nonverbals: dict[str, list[str]] = field(default_factory=dict)
    """``NonverbalOptions`` field -> the labels it governs; an absent field is a no-op"""


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
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    class Markup:
        """The markup dialect this TTS speaks, for the expressive pipeline: what the LLM is
        taught to write and how it is normalized and lowered before synthesis.

        Plugins opt in by overriding ``_provider_key``; the tables are the core's
        (``_core.Markup``).
        """

        def __init__(self, tts: TTS) -> None:
            self._tts = tts

        def _provider_key(self) -> str:
            """The provider's key into the markup tables, or "" for no markup."""
            return ""

        def _core(self) -> _core.Markup:
            return _core.Markup(self._provider_key())

        @property
        def info(self) -> MarkupInfo:
            """The queryable markup matrix for this voice."""
            return MarkupInfo(nonverbals=_core.supported_nonverbals(self._provider_key()))

        def llm_instructions(
            self, *, speech_steering: SpeechSteeringOptions | None = None
        ) -> str | None:
            """The instructions teaching the LLM this voice's markup, ``None`` without markup."""
            return self._core().llm_instructions(speech_steering=speech_steering)  # type: ignore[no-any-return]

        def normalize(self, text: str) -> str:
            """Fix common LLM markup mistakes (e.g. unclosed self-closing tags)."""
            return self._core().normalize(text)  # type: ignore[no-any-return]

        def convert(self, text: str) -> str:
            """Lower the framework's markup to the provider's native syntax."""
            return self._core().convert(text)  # type: ignore[no-any-return]

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
        self._markup = self.Markup(self)
        # whether expressive is active for the current turn, set by the framework before each
        # synthesis; a TTS tokenizing its own input batches larger chunks when it is
        self._expressive: bool = False

    @property
    def markup(self) -> Markup:
        """Access TTS markup capabilities (instructions for LLM, text stripping)."""
        return self._markup

    def _set_expressive(self, enabled: bool) -> None:
        """Mark whether expressive is active for this turn (set by the framework)."""
        self._expressive = enabled

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        """The model name; plugins override it."""
        return "unknown"

    @property
    def provider(self) -> str:
        """The provider name; plugins override it."""
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata used to label turn metrics emitted for this TTS instance."""
        return {"model_name": self.model, "model_provider": self.provider}

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

    def _synthesize_with_stream(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """``synthesize()`` for a TTS that only streams: one push, then the end of input."""
        return _ChunkedStreamFromStream(tts=self, input_text=text, conn_options=conn_options)

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


async def _drain_native(native: Any, output_emitter: AudioEmitter, *, stream: bool) -> None:
    """Forward a core synthesis (``_core.*TtsStream``) into ``output_emitter``."""
    started = not stream
    while (audio := await native.recv()) is not None:
        if not started:
            started = True
            output_emitter.start_segment(segment_id=audio["segment_id"] or shortuuid())
        if words := audio["timed_transcript"]:
            output_emitter.push_timed_transcript(words)
        output_emitter.push_frame(audio["frame"])


class _Synthesis:
    """What both stream kinds share: one attempt run against an ``AudioEmitter``; retries are
    the core's."""

    _tts: TTS
    _event_ch: aio.Chan[SynthesizedAudio]

    def _set_token_usage(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._tts.emit(
            "error",
            TTSError(
                timestamp=time.time(),
                label=self._tts._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    @abstractmethod
    async def _run(self, output_emitter: AudioEmitter) -> None: ...

    def _invalid(self, output_emitter: AudioEmitter) -> str | None: ...

    async def _attempt(self) -> None:
        output_emitter = AudioEmitter(label=self._tts.label, dst_ch=self._event_ch)
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if problem := self._invalid(output_emitter):
                raise APIError(problem)
        except APIError as e:
            # 499 (Client Closed Request) - close gracefully without raising
            if isinstance(e, APIStatusError) and e.status_code == 499:
                return
            self._emit_error(e, recoverable=False)
            raise
        finally:
            await output_emitter.aclose()


class ChunkedStream(_Synthesis, ABC):
    """Used by the non-streamed synthesize API, some providers support chunked http responses"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        # the full text is submitted to the provider at creation time
        self._started_time: float = time.perf_counter()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._input_tokens = 0
        self._output_tokens = 0
        self._acquire_time: float = 0.0
        self._connection_reused: bool = False
        self._synthesize_task = asyncio.create_task(self._attempt(), name="TTS._synthesize_task")
        self._synthesize_task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def input_text(self) -> str:
        return self._input_text

    def _invalid(self, output_emitter: AudioEmitter) -> str | None:
        return _core.validate_synthesis(self._input_text, output_emitter.pushed_duration(), None)  # type: ignore[no-any-return]

    @property
    def done(self) -> bool:
        return self._synthesize_task.done()

    @property
    def exception(self) -> BaseException | None:
        return self._synthesize_task.exception()

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        return rtc.combine_audio_frames([ev.frame async for ev in self])

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.cancel_and_wait(self._synthesize_task)
        self._event_ch.close()

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (exc := self._synthesize_task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
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


class _ChunkedStreamFromStream(ChunkedStream):
    """``synthesize()`` over a TTS that only streams."""

    async def _run(self, output_emitter: AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        async with self._tts.stream(conn_options=self._conn_options) as stream:
            stream.push_text(self._input_text)
            stream.end_input()
            async for ev in stream:
                output_emitter.push(ev.frame.data.tobytes())
                if timed_transcripts := ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(timed_transcripts)
        output_emitter.flush()


class SynthesizeStream(_Synthesis, ABC):
    class _FlushSentinel: ...

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self._input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._task = asyncio.create_task(self._attempt(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._started_time: float = 0
        self._pushed_text: str = ""
        self._segment_open = False
        self._num_segments = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._acquire_time: float = 0.0
        self._connection_reused: bool = False

    def _invalid(self, output_emitter: AudioEmitter) -> str | None:
        return _core.validate_synthesis(  # type: ignore[no-any-return]
            self._pushed_text,
            output_emitter.pushed_duration(),
            (self._num_segments, output_emitter.num_segments),
        )

    def _mark_started(self) -> None:
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if not token or self._input_ch.closed:
            return
        if not self._segment_open:
            if self._num_segments >= 1:
                logger.warning(
                    "SynthesizeStream: handling multiple segments in a single instance is "
                    "deprecated. Please create a new SynthesizeStream instance for each segment."
                )
                return
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += token
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self._input_ch.closed:
            return
        self._segment_open = False
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

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if self._started_time:
            val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
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
        self._provider_request_ids: list[str] = []  # deduped provider-known segment ids

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
            bytes
            | rtc.AudioFrame
            | AudioEmitter._FlushSegment
            | AudioEmitter._StartSegment
            | AudioEmitter._EndSegment
            | TimedString
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

        self._note_provider_request_id(segment_id)
        return self.__start_segment(segment_id=segment_id)

    def _note_provider_request_id(self, context_id: str) -> None:
        """Record a provider-known id for this stream (plugins call it once they learn one)."""
        if context_id and context_id not in self._provider_request_ids:
            self._provider_request_ids.append(context_id)

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

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Forward an already-framed chunk as-is, skipping the progressive re-chunking of
        :meth:`push`. For adapters wrapping a TTS whose emitter already framed the audio."""
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        self._write_ch.send_nowait(frame)

    def push_timed_transcript(self, delta_text: TimedString | list[TimedString]) -> None:
        if not self._started:
            raise RuntimeError("AudioEmitter isn't started")

        if self._write_ch.closed:
            return

        if isinstance(delta_text, list):
            for text in delta_text:
                self._write_ch.send_nowait(text)
        else:
            self._write_ch.send_nowait(delta_text)

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
        timed_transcripts: list[TimedString] = []

        flush_timer: asyncio.TimerHandle | None = None
        sent_start: float | None = None
        sent_duration: float = 0.0
        event_loop = asyncio.get_event_loop()

        def _send_audio(ev: SynthesizedAudio, *, flush_if_delayed: bool = False) -> None:
            nonlocal sent_start, sent_duration, flush_timer

            self._dst_ch.send_nowait(ev)
            if sent_start is None:
                sent_start = event_loop.time()
            sent_duration += ev.frame.duration

            if flush_timer is not None:
                flush_timer.cancel()

            def _flush() -> None:
                self.flush()
                logger.debug("flush audio emitter due to slow audio generation")

            if flush_if_delayed and sent_duration > 0.15:
                # force flush the buffer if the audio comes slower than realtime.
                # skip during the initial progressive ramp-up where sent_duration
                # is too small for a meaningful slow-generation check.
                delay = sent_duration - (event_loop.time() - sent_start) - 0.02
                if delay > 0:
                    flush_timer = event_loop.call_later(delay, _flush)

        # Number of samples held back in last_frame so we can tag is_final
        # on the very last audio of a segment.  10 ms is small enough to be
        # imperceptible but avoids holding a full-sized frame.
        _TAIL_SAMPLES = self._sample_rate * 10 // 1000

        def _split_tail(frame: rtc.AudioFrame) -> tuple[rtc.AudioFrame | None, rtc.AudioFrame]:
            """Split *frame* into (head, tail) where tail is exactly _TAIL_SAMPLES.

            If the frame is too small to split, returns (None, frame).
            """
            if frame.samples_per_channel <= _TAIL_SAMPLES:
                return None, frame
            head_samples = frame.samples_per_channel - _TAIL_SAMPLES
            # frame.data is a memoryview of int16 — slice by sample * num_channels
            split_idx = head_samples * frame.num_channels
            head = rtc.AudioFrame(
                data=frame.data[:split_idx],
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=head_samples,
            )
            tail = rtc.AudioFrame(
                data=frame.data[split_idx:],
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=_TAIL_SAMPLES,
            )
            return head, tail

        def _do_send(frame: rtc.AudioFrame, *, is_final: bool) -> None:
            """Send a frame downstream and update bookkeeping."""
            nonlocal segment_ctx, timed_transcripts
            assert segment_ctx is not None

            frame.userdata[USERDATA_TIMED_TRANSCRIPT] = timed_transcripts
            timed_transcripts = []
            _send_audio(
                SynthesizedAudio(
                    frame=frame,
                    request_id=self._request_id,
                    segment_id=segment_ctx.segment_id,
                    is_final=is_final,
                ),
                flush_if_delayed=not is_final,
            )
            segment_ctx.audio_duration += frame.duration
            self._audio_durations[-1] += frame.duration

        def _emit_frame(frame: rtc.AudioFrame | None = None, *, is_final: bool = False) -> None:
            nonlocal last_frame, segment_ctx, timed_transcripts
            assert segment_ctx is not None

            if is_final:
                # end of segment — flush everything
                if last_frame is not None and frame is not None:
                    # merge last_frame + frame, send as final
                    combined = rtc.combine_audio_frames([last_frame, frame])
                    _do_send(combined, is_final=True)
                    last_frame = None
                elif last_frame is not None:
                    _do_send(last_frame, is_final=True)
                    last_frame = None
                elif frame is not None:
                    _do_send(frame, is_final=True)
                elif segment_ctx.audio_duration > 0:
                    # no frame but segment had audio — send a tiny empty marker
                    # without updating duration tracking (it's synthetic silence)
                    marker = rtc.AudioFrame(
                        data=b"\0\0" * (self._sample_rate // 100 * self._num_channels),
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                        samples_per_channel=self._sample_rate // 100,
                    )
                    marker.userdata[USERDATA_TIMED_TRANSCRIPT] = timed_transcripts
                    timed_transcripts = []
                    _send_audio(
                        SynthesizedAudio(
                            frame=marker,
                            request_id=self._request_id,
                            segment_id=segment_ctx.segment_id,
                            is_final=True,
                        ),
                        flush_if_delayed=False,
                    )
                return

            if frame is None:
                return

            # Normal (non-final) frame: send as much as possible, hold back
            # only a small tail so we can mark the last audio as is_final.
            if last_frame is not None:
                # combine previous tail with new frame before splitting
                combined = rtc.combine_audio_frames([last_frame, frame])
            else:
                combined = frame

            head, tail = _split_tail(combined)
            if head is not None:
                _do_send(head, is_final=False)
            last_frame = tail

        def _flush_frame() -> None:
            nonlocal last_frame, segment_ctx, timed_transcripts
            nonlocal flush_timer, sent_start, sent_duration
            assert segment_ctx is not None

            if last_frame is None:
                return

            last_frame.userdata[USERDATA_TIMED_TRANSCRIPT] = timed_transcripts
            _send_audio(
                SynthesizedAudio(
                    frame=last_frame,
                    request_id=self._request_id,
                    segment_id=segment_ctx.segment_id,
                    is_final=False,  # flush isn't final
                ),
                flush_if_delayed=False,  # don't flush again before new frames are pushed
            )
            timed_transcripts = []
            segment_ctx.audio_duration += last_frame.duration
            self._audio_durations[-1] += last_frame.duration

            last_frame = None
            # reset sent duration after flush
            sent_start = None
            sent_duration = 0.0
            if flush_timer is not None:
                flush_timer.cancel()
                flush_timer = None

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
                        progressive=True,
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
                if isinstance(data, TimedString):
                    timed_transcripts.append(data)
                    continue

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
                    if isinstance(data, rtc.AudioFrame):
                        _emit_frame(data)
                    elif isinstance(data, bytes):
                        if audio_byte_stream is None:
                            audio_byte_stream = audio.AudioByteStream(
                                sample_rate=self._sample_rate,
                                num_channels=self._num_channels,
                                samples_per_channel=int(
                                    self._sample_rate // 1000 * self._frame_size_ms
                                ),
                                progressive=True,
                            )

                        for f in audio_byte_stream.push(data):
                            _emit_frame(f)
                    elif isinstance(data, AudioEmitter._FlushSegment):
                        if audio_byte_stream:
                            for f in audio_byte_stream.flush():
                                _emit_frame(f)
                            # More bytes can follow this flush. Keep any partial PCM sample.
                            audio_byte_stream.reset_progressive()
                        _flush_frame()

                    elif isinstance(data, AudioEmitter._EndSegment):
                        if audio_byte_stream:
                            for f in audio_byte_stream.flush():
                                _emit_frame(f)
                            if audio_byte_stream.buffered_duration > 0:
                                logger.warning(
                                    "incomplete PCM sample at end of segment, "
                                    "discarding trailing bytes",
                                    extra={"tts": self._label, "request_id": self._request_id},
                                )

                        _emit_frame(is_final=True)
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
                        if isinstance(data, AudioEmitter._FlushSegment):
                            # don't tear the decoder down here. flush_if_delayed
                            # can fire mid-stream while a stateful codec
                            # (WAV/OGG/MP3) is in the middle of a file; ending
                            # input would discard the parser, and the next
                            # bytes — a pure PCM/Opus packet continuation
                            # without a fresh container header — would fail to
                            # parse against a freshly-created decoder. The only
                            # purpose of FlushSegment here is to release the
                            # held-back tail so a slow upstream doesn't starve
                            # the consumer.
                            _flush_frame()

                        elif isinstance(data, AudioEmitter._EndSegment) and segment_ctx:
                            if audio_decoder:
                                audio_decoder.end_input()
                                await decode_atask
                            _emit_frame(is_final=True)
                            audio_decoder = segment_ctx = audio_byte_stream = last_frame = None
                        else:
                            logger.warning("unknown data type: %s", type(data))

        finally:
            if flush_timer is not None:
                flush_timer.cancel()

            if audio_decoder and decode_atask:
                await audio_decoder.aclose()
                await aio.cancel_and_wait(decode_atask)
