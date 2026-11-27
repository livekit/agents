from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, unique
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc
from livekit.agents.metrics.base import Metadata

from .._exceptions import APIConnectionError, APIError
from ..language import LanguageCode
from ..log import logger
from ..metrics import STTMetrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    TimedString,
)
from ..utils import AudioBuffer, aio, is_given
from ..utils.audio import calculate_audio_duration
from .clock import _ConnectionClock, _InputClock

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata
    from ..voice.events import ConversationItemAddedEvent


@unique
class SpeechEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""  # noqa: E501
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    PREFLIGHT_TRANSCRIPT = "preflight_transcript"
    """preflight transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change. This is different from final transcript in that
    the same transcript may still be updated; but it is stable enough to be used for
    preemptive generation"""
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change"""
    RECOGNITION_USAGE = "recognition_usage"
    """usage event, emitted periodically to indicate usage metrics"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking.
    alternatives are optional and provider-specific; they may contain only the latest segment.
    build the complete transcript from FINAL_TRANSCRIPT events"""


@dataclass
class SpeechData:
    language: LanguageCode
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0  # [0, 1]
    speaker_id: str | None = None
    is_primary_speaker: bool | None = None
    words: list[TimedString] | None = None
    source_languages: list[LanguageCode] | None = None
    """the source languages spoken by the user. populated by STT services that support translation,
    where `language` holds the target language and `source_languages` holds the original spoken language(s),
    or by multi-language detection services where `language` holds the dominant language and
    `source_languages` holds all detected languages sorted by prevalence.
    may contain multiple entries when a single utterance spans multiple source languages."""
    source_texts: list[str] | None = None
    """the original transcription segments in the source language(s), when translation is active.
    each entry corresponds to the same-indexed entry in `source_languages`."""
    target_languages: list[LanguageCode] | None = None
    """the target language(s) produced by a translation-capable STT service, one entry per
    consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    `source_texts` on the source side: `language` holds the dominant (first) target language
    and `target_languages` carries the fine-grained per-run breakdown.
    populated when translation is active; None otherwise."""
    target_texts: list[str] | None = None
    """the translated transcription segments in the target language(s).
    each entry corresponds to the same-indexed entry in `target_languages`."""
    metadata: dict[str, Any] | None = None
    """optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    plugins may populate this with provider-specific data that doesn't map to standard fields."""

    def __post_init__(self) -> None:
        if not isinstance(self.language, LanguageCode) and isinstance(self.language, str):
            self.language = LanguageCode(self.language)
        if self.source_languages is not None:
            self.source_languages = [
                LanguageCode(lang)
                if not isinstance(lang, LanguageCode) and isinstance(lang, str)
                else lang
                for lang in self.source_languages
            ]
        if self.target_languages is not None:
            self.target_languages = [
                LanguageCode(lang)
                if not isinstance(lang, LanguageCode) and isinstance(lang, str)
                else lang
                for lang in self.target_languages
            ]


@dataclass
class RecognitionUsage:
    audio_duration: float
    """Incremental audio duration/usage in seconds"""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None
    speech_start_time: float | None = None
    """Speech onset wall-clock time for event consumers.

    When ``requires_timestamp_translation`` is true, supply connection-relative
    seconds. The event channel translates them to wall-clock time.
    """
    created_at: float = field(default_factory=lambda: time.time())
    """Wall-clock time when this event was created."""
    speech_end_time: float | None = None
    """Time when the recognized speech ended, when known. Same clocks as
    ``speech_start_time``."""


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool
    diarization: bool = False
    aligned_transcript: Literal["word", "chunk", False] = False
    offline_recognize: bool = True
    """Whether the STT supports batch recognition via recognize() method"""
    keyterms: bool = False
    """Whether the STT supports keyterm prompting"""
    chat_context: bool = False
    """Whether the STT can natively consume conversation context (see STT._push_conversation_item)"""
    manual_flush: bool = False
    """Whether the STT supports flushing the stream to finalize a segment.

    When enabled, manual turn commits request a flush. With a turn detector, automatic
    flushing occurs immediately for positive or unavailable EOT predictions, or halfway
    through the maximum endpointing delay for negative predictions. Without a turn
    detector, VAD end-of-speech requests the flush. Text-based EOT uses final transcripts
    only, so flushing precedes its first prediction if no final transcript is available.
    Disable this for custom STT nodes that buffer audio and manage flushing themselves.
    """


class STTError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class STT(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    def __init__(
        self, *, capabilities: STTCapabilities, requires_timestamp_translation: bool = True
    ) -> None:
        """
        Args:
            capabilities: The features that this STT supports.
            requires_timestamp_translation: Whether the streams of this STT emit event
                timestamps relative to their connection. If true, the output channel
                translates those timestamps to the session timeline. Set it to false
                when the streams emit session timestamps themselves.
        """
        super().__init__()
        self._capabilities = capabilities
        self._requires_timestamp_translation = requires_timestamp_translation
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._recognize_metrics_needed = True
        self._keyterms_unsupported_warned = False
        self._chat_context_unsupported_warned = False

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        """Get the model name/identifier for this STT instance.

        Returns:
            The model name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their model information.
        """
        return "unknown"

    @property
    def provider(self) -> str:
        """Get the provider name/identifier for this STT instance.

        Returns:
            The provider name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their provider information.
        """
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata used to label turn metrics emitted for this STT instance."""
        return {"model_name": self.model, "model_provider": self.provider}

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    @property
    def requires_timestamp_translation(self) -> bool:
        """Whether the event channel must translate timestamps to the session timeline."""
        return self._requires_timestamp_translation

    @abstractmethod
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent: ...

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        for i in range(conn_options.max_retry + 1):
            try:
                start_time = time.perf_counter()
                event = await self._recognize_impl(
                    buffer, language=language, conn_options=conn_options
                )
                if self._recognize_metrics_needed:
                    duration = time.perf_counter() - start_time
                    stt_metrics = STTMetrics(
                        request_id=event.request_id,
                        timestamp=time.time(),
                        duration=duration,
                        label=self._label,
                        audio_duration=calculate_audio_duration(buffer),
                        streamed=False,
                        metadata=Metadata(
                            model_name=self.model,
                            model_provider=self.provider,
                        ),
                    )
                    self.emit("metrics_collected", stt_metrics)
                return event

            except APIError as e:
                retry_interval = conn_options._interval_for_retry(i)
                if conn_options.max_retry == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif i == conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to recognize speech after {conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    logger.warning(
                        f"failed to recognize speech: {e}, retrying in {retry_interval}s",
                        extra={
                            "stt": self._label,
                            "attempt": i + 1,
                            "streamed": False,
                        },
                    )

                await asyncio.sleep(retry_interval)

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

        raise RuntimeError("unreachable")

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            STTError(
                timestamp=time.time(),
                label=self._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        """Set the framework-managed keyterms (session config + auto-detection).

        Internal hook called by the framework, kept separate from the user's own keyterms
        (constructor / ``update_options``). Plugins that support keyterms override this to
        store the session set and apply it merged with the user keyterms.
        """
        if not self._capabilities.keyterms:
            if not self._keyterms_unsupported_warned:
                self._keyterms_unsupported_warned = True
                logger.warning(
                    "keyterms are not supported by this STT, ignoring keyterms update",
                    extra={"stt": self._label},
                )
            return

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        """Feed a new conversation turn to the STT to bias recognition (context carryover).

        Plugins with native context support set ``STTCapabilities.chat_context`` and override
        this to forward the item to their provider's carryover field.
        """
        if not self._capabilities.chat_context:
            if not self._chat_context_unsupported_warned:
                self._chat_context_unsupported_warned = True
                logger.warning(
                    "chat context is not supported by this STT, ignoring chat context update",
                    extra={"stt": self._label},
                )
            return

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"  # noqa: E501
        )

    async def aclose(self) -> None:
        """Close the STT, and every stream/requests associated with it"""
        ...

    async def __aenter__(self) -> STT:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def prewarm(self) -> None:
        """Pre-warm connection to the STT service"""
        pass


@runtime_checkable
class SpeakerContext(Protocol):
    """Protocol for STT context models that can generate LLM instructions.

    STT plugins that provide speaker metadata should implement this protocol
    on their context model so the expressive system can auto-inject
    speaker context into the LLM.
    """

    def to_instructions(self) -> str: ...


class RecognizeStream(ABC):
    class _FlushSentinel:
        """Sentinel to mark when it was flushed"""

        pass

    class _InputChan(aio.Chan[rtc.AudioFrame | _FlushSentinel]):
        """Report the duration of each audio frame when it is pushed and when it is read."""

        def __init__(
            self,
            *,
            on_audio_enqueued: Callable[[float], None],
            on_audio_dequeued: Callable[[float], None],
        ) -> None:
            super().__init__()
            self._on_audio_enqueued = on_audio_enqueued
            self._on_audio_dequeued = on_audio_dequeued

        def send_nowait(self, value: rtc.AudioFrame | RecognizeStream._FlushSentinel) -> None:
            super().send_nowait(value)
            if isinstance(value, rtc.AudioFrame) and value.duration > 0:
                self._on_audio_enqueued(value.duration)

        def recv_nowait(self) -> rtc.AudioFrame | RecognizeStream._FlushSentinel:
            value = super().recv_nowait()
            if isinstance(value, rtc.AudioFrame) and value.duration > 0:
                self._on_audio_dequeued(value.duration)
            return value

    class _EventChan(aio.Chan[SpeechEvent]):
        """Translate timestamps before enqueueing events.

        ``translate_event_timestamps`` returns a copy of an event on the session timeline.
        """

        def __init__(
            self, *, translate_event_timestamps: Callable[[SpeechEvent], SpeechEvent] | None
        ) -> None:
            super().__init__()
            self._translate_event_timestamps = translate_event_timestamps

        def send_nowait(self, value: SpeechEvent) -> None:
            """Queue ``value``, translating its timestamps when a callback is set."""
            if self._translate_event_timestamps is not None:
                value = self._translate_event_timestamps(value)
            super().send_nowait(value)

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ):
        """
        Args:
        sample_rate : int or None, optional
            The desired sample rate for the audio input.
            If specified, the audio input will be automatically resampled to match
            the given sample rate before being processed for Speech-to-Text.
            If not provided (None), the input will retain its original sample rate.
        """
        self._stt = stt
        self._conn_options = conn_options
        self._init_channels(translate_timestamps=stt.requires_timestamp_translation)

        self._tee = aio.itertools.tee(self._event_ch, 2)
        self._event_aiter, monitor_aiter = self._tee
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="STT._metrics_task"
        )

        self._num_retries = 0
        self._on_provider_progress: Callable[[], None] | None = None
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None

        self._start_time: float = time.time()
        self._context: BaseModel | None = None

    def _init_channels(self, *, translate_timestamps: bool) -> None:
        """Create the input and event channels and the clocks they report to."""
        self._input_clock = _InputClock()
        self._connection_clock = _ConnectionClock(self._input_clock, 0.0)
        self._connection_clock_restart_pending = True

        self._input_ch = RecognizeStream._InputChan(
            on_audio_enqueued=self._on_audio_enqueued, on_audio_dequeued=self._on_audio_dequeued
        )
        self._event_ch = RecognizeStream._EventChan(
            translate_event_timestamps=self._translate_event_timestamps
            if translate_timestamps
            else None
        )

    def _on_audio_enqueued(self, duration: float) -> None:
        self._input_clock.on_audio_enqueued(duration, time.perf_counter(), self._connection_clock)

    def _on_audio_dequeued(self, duration: float) -> None:
        if self._connection_clock_restart_pending:
            self._connection_clock_restart_pending = False
            self._connection_clock = self._input_clock.start_connection()
        self._input_clock.on_audio_dequeued(duration)

    def _translate_event_timestamps(self, event: SpeechEvent) -> SpeechEvent:
        return self._connection_clock.translate_event_timestamps(event)

    def _restart_connection_clock(self, *, replayed_duration: float = 0.0) -> None:
        """Start the next connection's clock when its first frame is read.

        Keep the previous connection's mapping until then. ``replayed_duration`` is
        audio already read from the input that the next connection is sent first; its
        clock then starts now, that many seconds before the next frame.
        """
        if replayed_duration > 0:
            self._connection_clock_restart_pending = False
            self._connection_clock = self._input_clock.start_connection(replayed_duration)
            return
        self._connection_clock_restart_pending = True

    @property
    def context(self) -> BaseModel | None:
        """Persistent speaker metadata updated by the STT plugin during recognition.

        STT plugins set this to a ``BaseModel`` instance (e.g. a voice profile with
        emotion, gender, etc.) as they learn about the speaker over the life of the
        stream.  Accessible via ``agent.audio_recognition.stt_context``.
        """
        return self._context

    @context.setter
    def context(self, value: BaseModel | None) -> None:
        self._context = value

    @property
    def start_time_offset(self) -> float:
        """Session position of the current connection's first audio frame, in seconds.

        Setting this value declares that the session timeline began that many
        seconds ago. When unset, it begins with the stream's first audio. Each new
        connection re-anchors the offset when it reads its first frame.
        """
        return self._connection_clock.start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._input_clock.set_session_timeline(value)

    @property
    def start_time(self) -> float:
        """Wall-clock time of stream creation, reset when each ``_run`` attempt starts.

        Timestamp mapping does not use this value; it follows ``start_time_offset``
        and the input audio. Plugins can set it for their own bookkeeping.
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time must be non-negative")
        self._start_time = value

    def _report_connection_acquired(self, acquire_time: float, connection_reused: bool) -> None:
        """Report connection timing as an STTMetrics event with zero usage."""
        self._stt.emit(
            "metrics_collected",
            STTMetrics(
                request_id="",
                timestamp=time.time(),
                duration=0.0,
                label=self._stt._label,
                audio_duration=0.0,
                streamed=True,
                acquire_time=acquire_time,
                connection_reused=connection_reused,
                metadata=Metadata(model_name=self._stt.model, model_provider=self._stt.provider),
            ),
        )

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        max_retries = self._conn_options.max_retry

        while self._num_retries <= max_retries:
            try:
                self._start_time = time.time()
                return await self._run()
            except APIError as e:
                # _run has stopped reading, so the next frame read is the next connection's
                self._restart_connection_clock()
                if not e.retryable or max_retries == 0:
                    self._emit_error(e, recoverable=False)
                    raise
                elif self._num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to recognize speech after {self._num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"failed to recognize speech: {e}, retrying in {retry_interval}s",
                        extra={
                            "stt": self._stt._label,
                            "attempt": self._num_retries,
                            "streamed": True,
                        },
                    )
                    await asyncio.sleep(retry_interval)

                self._num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _mark_provider_progress(self) -> None:
        """Reset recovery retries after a valid provider processing event.

        Call at the receive/parse boundary for speech, turn, transcription, or
        audio-processing events. Setup, errors, heartbeats, and local events do
        not indicate provider progress.
        """
        self._num_retries = 0
        if self._on_provider_progress is not None:
            self._on_provider_progress()

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt.emit(
            "error",
            STTError(
                timestamp=time.time(),
                label=self._stt._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SpeechEvent]) -> None:
        """Task used to collect metrics"""

        async for ev in event_aiter:
            if ev.type == SpeechEventType.RECOGNITION_USAGE:
                assert ev.recognition_usage is not None, (
                    "recognition_usage must be provided for RECOGNITION_USAGE event"
                )

                stt_metrics = STTMetrics(
                    request_id=ev.request_id,
                    timestamp=time.time(),
                    duration=0.0,
                    label=self._stt._label,
                    audio_duration=ev.recognition_usage.audio_duration,
                    input_tokens=ev.recognition_usage.input_tokens,
                    output_tokens=ev.recognition_usage.output_tokens,
                    streamed=True,
                    metadata=Metadata(
                        model_name=self._stt.model, model_provider=self._stt.provider
                    ),
                )

                self._stt.emit("metrics_collected", stt_metrics)
            elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                # Preserve recovery resets for custom providers that emit only transcripts.
                self._mark_provider_progress()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized"""
        self._check_input_not_ended()
        self._check_not_closed()

        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")

        self._pushed_sr = frame.sample_rate

        if self._needed_sr and self._needed_sr != frame.sample_rate:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

        if self._resampler:
            frames = self._resampler.push(frame)
            for frame in frames:
                self._input_ch.send_nowait(frame)
        else:
            self._input_ch.send_nowait(frame)

    @property
    def manual_flush(self) -> bool:
        """Whether this stream finalizes a turn only when flushed.

        Defaults to the STT's current capability. A stream whose need is fixed when it is
        created overrides this, so a later STT option change does not stop its flushes.
        """
        return self._stt.capabilities.manual_flush

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()

        if self._resampler:
            for frame in self._resampler.flush():
                self._input_ch.send_nowait(frame)

        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

        if self._metrics_task is not None:
            await aio.cancel_and_wait(self._metrics_task)

        await self._tee.aclose()

    async def __anext__(self) -> SpeechEvent:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

        return val

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __aenter__(self) -> RecognizeStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


SpeechStream = RecognizeStream  # deprecated alias
