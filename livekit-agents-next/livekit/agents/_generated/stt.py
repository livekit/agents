# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/stt/stt.py

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum, unique
from inspect import get_annotations
from types import TracebackType
from typing import Annotated, Generic, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing_extensions import TypeVar

from livekit import rtc

from .. import _bindings, _core
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, is_given
from .chat_items import MetricsMetadata
from .language import LanguageCode
from .transcript import TranscriptChunk
from .types import ProviderId
from .usage import TokenUsage

# Record declarations:
#     rust/crates/lk-agents-abi/src/stt/events.rs
#     rust/crates/lk-agents-abi/src/stt/stt.rs


@unique
class SpeechEventType(StrEnum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT
    """
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    PREFLIGHT_TRANSCRIPT = "preflight_transcript"
    """preflight transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change. This is different from final transcript in that
    the same transcript may still be updated; but it is stable enough to be used for
    preemptive generation
    """
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change
    """
    RECOGNITION_USAGE = "recognition_usage"
    """usage event, emitted periodically to indicate usage metrics"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


@dataclass
class SpeechData:
    language: LanguageCode
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    speaker_id: str | None = None
    words: list[TranscriptChunk] | None = None
    source_languages: list[LanguageCode] | None = None
    """the source languages spoken by the user. populated by STT services that support translation,
    where `language` holds the target language and `source_languages` holds the original spoken language(s),
    or by multi-language detection services where `language` holds the dominant language and
    `source_languages` holds all detected languages sorted by prevalence.
    may contain multiple entries when a single utterance spans multiple source languages.
    """
    source_texts: list[str] | None = None
    """the original transcription segments in the source language(s), when translation is active.
    each entry corresponds to the same-indexed entry in `source_languages`.
    """
    target_languages: list[LanguageCode] | None = None
    """the target language(s) produced by a translation-capable STT service, one entry per
    consecutive same-language run, parallel to `target_texts`. mirrors `source_languages` /
    `source_texts` on the source side: `language` holds the dominant (first) target language
    and `target_languages` carries the fine-grained per-run breakdown.
    populated when translation is active; None otherwise.
    """
    target_texts: list[str] | None = None
    """the translated transcription segments in the target language(s).
    each entry corresponds to the same-indexed entry in `target_languages`.
    """
    metadata: dict[str, JsonValue] | None = None
    """optional plugin-specific metadata (e.g. voice profile, provider diagnostics).
    plugins may populate this with provider-specific data that doesn't map to standard fields.
    """

    def __post_init__(self) -> None:
        from .. import _bindings

        _bindings.speech_data_post_init(self)


@dataclass
class RecognitionUsage:
    audio_duration: float | None = None
    """Incremental processed audio in seconds, when the provider measures it."""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    """Token-based billing; zero when the provider reports no tokens."""


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None
    speech_start_time: float | None = None
    """server-reported wall-clock time of speech onset, when the provider sends
    a separate speech-start signal carrying onset timing.
    """
    created_at: float = field(default_factory=time.time)
    """Wall-clock time when this event was created."""
    speech_end_time: float | None = None
    """Wall-clock time when the recognized speech ended, when known."""


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


class STTError(BaseModel):
    """The public provider error event; its Python exception is retained but never serialized."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool


SpeechEventType.__module__ = "livekit.agents.stt"
SpeechData.__annotations__ = get_annotations(SpeechData, eval_str=True)
SpeechData.__module__ = "livekit.agents.stt"
RecognitionUsage.__annotations__ = get_annotations(RecognitionUsage, eval_str=True)
RecognitionUsage.__module__ = "livekit.agents.stt"
SpeechEvent.__annotations__ = get_annotations(SpeechEvent, eval_str=True)
SpeechEvent.__module__ = "livekit.agents.stt"
STTCapabilities.__annotations__ = get_annotations(STTCapabilities, eval_str=True)
STTCapabilities.__module__ = "livekit.agents.stt"
STTError.__module__ = "livekit.agents.stt"


TEvent = TypeVar("TEvent", default=Never, contravariant=True)

# Events the reader may leave unread before the core's input waits for it.
_EVENT_BACKLOG = 64


class STT(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    """A speech-to-text provider.

    Plugins implement ``_recognize_impl`` for batch recognition, and override ``stream()`` when the
    service recognizes live audio.
    """

    def __init__(self, *, capabilities: STTCapabilities) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._capabilities = capabilities

    def _core_leg(self) -> object:
        """The Python provider implementation used by the native runtime's declared proxy."""
        return self

    @property
    def label(self) -> str:
        """The name logs and metrics give this instance: its class path."""
        return self._label

    @property
    def capabilities(self) -> STTCapabilities:
        """What this instance supports, fixed at construction."""
        return self._capabilities

    @property
    def model(self) -> str:
        """The model name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def provider(self) -> str:
        """The provider name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """The model and provider labelling this instance's turn metrics."""
        return {"model_name": self.model, "model_provider": self.provider}

    def prewarm(self) -> None:
        """Open the connection to the service early; a provider with nothing to warm ignores it."""

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        """Set the session's keyterms: its configured ones and those confirmed in the conversation,
        kept apart from the user's own. Providers with the ``keyterms`` capability override it.

        Args:
            keyterms: Every keyterm the session holds now.
        """

    def _push_conversation_item(self, role: str, text: str) -> None:
        """Feed a conversation turn to bias recognition. Providers with the ``chat_context``
        capability override it.

        Args:
            role: Who spoke the turn: ``"user"`` or ``"assistant"``.
            text: What was said.
        """

    @abstractmethod
    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent: ...

    async def recognize(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        """Recognize a recording, its attempts retried per ``conn_options``.

        Args:
            buffer: The audio to recognize.
            language: The language to recognize; the provider's default when not given.
            conn_options: The timeout and retries of the request.
        """
        return await _bindings.recognize(self, buffer, language=language, conn_options=conn_options)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        """Open a stream recognizing live audio. Providers with the ``streaming`` capability
        override it; any other is wrapped in a ``StreamAdapter``.

        Args:
            language: The language to recognize; the provider's default when not given.
            conn_options: The timeout and retries of the connection.
        """
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"  # noqa: E501
        )

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

    async def aclose(self) -> None:  # noqa: B027
        """Close the provider and everything it opened."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _core_instance_id(self) -> int | None:
        return id(self)

    def _core_label(self) -> str:
        return self.label

    def _core_capabilities(self) -> STTCapabilities:
        return self.capabilities

    def _core_id(self) -> ProviderId:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    def _core_prewarm(self) -> None:
        self.prewarm()

    def _core_update_keyterms(self, keyterms: list[str]) -> None:
        self._update_session_keyterms(keyterms)

    def _core_push_conversation_item(self, role: str, text: str) -> None:
        self._push_conversation_item(role, text)

    def _core_failed(self, error: Exception, recoverable: bool) -> None:
        self._emit_error(error, recoverable)

    def _core_conn_options(self) -> APIConnectOptions | None:
        return None

    async def _core_recognize(
        self,
        buffer: list[rtc.AudioFrame],
        language: str | None,
        conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._recognize_impl(
            buffer, language=NOT_GIVEN if language is None else language, conn_options=conn
        )

    async def _core_stream(
        self,
        language: str | None,
        start_time_offset: float | None,
        conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream | None:
        if not self.capabilities.streaming:
            return None
        out = self.stream(language=NOT_GIVEN if language is None else language, conn_options=conn)
        out._core_attempts.core = True
        if start_time_offset is not None:
            out.start_time_offset = start_time_offset
        return out


class RecognizeStream(ABC):
    """A live recognition: audio in through ``push_frame``, ``SpeechEvent`` out.

    Plugins implement ``_run``: one attempt at the recognition, reading ``input`` and handing
    each event to ``send``. The framework runs it under ``conn_options`` (docs/models.md
    "Provider attempts"): a retryable ``APIError`` reconnects by calling ``_run`` again, which
    reads the audio the failed attempt had not taken; transcripts already sent stay delivered.
    """

    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            stt: The provider this stream recognizes for.
            conn_options: The timeout and retries of the connection: a failed connection is
                reopened, the audio it had not taken read by the next.
            sample_rate: The rate the audio is resampled to before recognition; the input's own
                rate when not given.
        """
        self._stt = stt
        self._conn_options = conn_options
        self.input = aio.Chan[rtc.AudioFrame | RecognizeStream.Flush](
            maxsize=256, _on_receive=self._input_received, _on_eof=self._input_ended
        )
        """The audio ``_run`` reads, a ``Flush`` at the end of each segment."""
        self._event_ch = aio.Chan[SpeechEvent](_on_receive=lambda _: self._event_read.set())
        self._event_read = asyncio.Event()
        self._start_time = time.time()
        self._start_time_offset = 0.0
        self._context: BaseModel | None = None
        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None
        self._core_attempts: _core.ProviderAttempts = _core.ProviderAttempts(
            "stt_stream", conn_options, self
        )
        self._core_attempt_task: asyncio.Task[None] | None = None
        self._core_cancelled = False
        self._attempt_started = 0.0
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._close_events())

    @property
    def context(self) -> BaseModel | None:
        """Speaker metadata the plugin updates as it learns about the speaker."""
        return self._context

    @context.setter
    def context(self, value: BaseModel | None) -> None:
        self._context = value

    @property
    def start_time_offset(self) -> float:
        """Where this stream's timestamps sit on the caller's input timeline, in seconds."""
        return self._start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._start_time_offset = value

    @property
    def start_time(self) -> float:
        """Wall-clock anchor of the stream; plugins may move it to when the first audio reached
        the provider, so stream-relative times convert accurately."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time must be non-negative")
        self._start_time = value

    @abstractmethod
    async def _run(self) -> None:
        """One attempt: connect, read ``input`` until it ends and ``send`` each event.

        Called once per attempt, in a fresh task; a retry calls it again with ``input`` still
        holding the audio not yet read, and ``start_time``/``start_time_offset`` moved on so
        timestamps stay linear. Raise ``APIError`` (``retryable=True`` for transient failures)
        to fail the attempt. Cancellation means a deadline passed or the stream was closed:
        clean up and let it propagate.
        """

    def _input_received(self, item: rtc.AudioFrame | RecognizeStream.Flush) -> None:
        if isinstance(item, self.Flush):
            self._core_attempts.expect_response()

    def _input_ended(self) -> None:
        self._core_attempts.input_ended()

    async def _main_task(self) -> None:
        self._attempt_started = time.time()
        # run() starts from the task, after the framework set ``core``
        await _bindings.run_attempts(self, self._core_attempts)

    async def _core_attempt(self, retry: bool) -> None:
        _bindings.attempt_started(self)
        # each attempt moves the timeline on by the time the last one ran, so a reconnected
        # stream's timestamps stay linear
        now = time.time()
        self._start_time_offset += now - self._attempt_started
        self._start_time = self._attempt_started = now
        await self._run()

    def _core_produced(self) -> bool:
        return False

    def _core_retrying(self, error: Exception) -> None:
        self._stt._emit_error(error, True)

    def _core_failed(self, error: Exception) -> None:
        self._stt._emit_error(error, False)

    def _close_events(self) -> None:
        self._event_ch.close()
        self._event_read.set()

    def _resampled(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        self._check_not_closed()
        self._check_input_not_ended()
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")
        self._pushed_sr = frame.sample_rate
        if self._needed_sr and self._needed_sr != frame.sample_rate and not self._resampler:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._needed_sr, quality=rtc.AudioResamplerQuality.HIGH
            )
        return self._resampler.push(frame) if self._resampler else [frame]

    def _segment_end(self) -> list[rtc.AudioFrame | RecognizeStream.Flush]:
        self._check_not_closed()
        self._check_input_not_ended()
        tail = self._resampler.flush() if self._resampler else []
        return [*tail, self.Flush()]

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized. Raises ``ChanFull`` while ``input`` holds 256 entries."""
        for f in self._resampled(frame):
            self.input.send_nowait(f)

    def flush(self) -> None:
        """Mark the end of the current segment."""
        for item in self._segment_end():
            self.input.send_nowait(item)

    def end_input(self) -> None:
        """Mark the end of input: no more audio will be pushed."""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately."""
        self.input.close()
        await aio.cancel_and_wait(self._task, timeout=self._core_attempts.cleanup_timeout)
        self._event_ch.close()

    async def __anext__(self) -> SpeechEvent:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if self._task.done() and not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        return val

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self

    def send(self, event: SpeechEvent) -> None:
        """Hand ``event`` to the reader."""
        self._core_attempts.observe(event)
        stt = self._stt
        record, _ = _core.stt_stream_metrics(event, time.time(), stt.label, stt.model, stt.provider)
        _bindings.emit_metrics(self._stt, record)
        self._event_ch.send_nowait(event)

    @property
    def closed(self) -> bool:
        """Whether the stream is closed: its reader takes nothing more."""
        return self._event_ch.closed

    def _check_not_closed(self) -> None:
        if self._task.done() and not self._task.cancelled() and (error := self._task.exception()):
            raise error
        if self.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self.input.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _core_push(self, frame: rtc.AudioFrame) -> None:
        # an unread backlog of events holds the input, so the core's queue keeps the audio
        while self._event_ch.qsize() >= _EVENT_BACKLOG and not self._event_ch.closed:
            self._event_read.clear()
            await self._event_read.wait()
        for f in self._resampled(frame):
            await self.input.send(f)

    async def _core_flush(self) -> None:
        if not self.input.closed:
            for item in self._segment_end():
                await self.input.send(item)

    async def _core_end_input(self) -> None:
        if not self.input.closed:
            await self._core_flush()
            self.input.close()




# The public module identity is assigned after all classes are built.
STT.__module__ = "livekit.agents.stt"
RecognizeStream.__module__ = "livekit.agents.stt"
