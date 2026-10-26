from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIConnectionError, APIError
from .._generated.stt import RecognitionUsage, SpeechData, SpeechEventType, STTCapabilities
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import AudioBuffer, aio, is_given

if TYPE_CHECKING:
    from ..llm.chat_context import MetricsMetadata
    from ..voice.events import ConversationItemAddedEvent

__all__ = [
    "STT",
    "STTCapabilities",
    "STTError",
    "RecognitionUsage",
    "RecognizeStream",
    "SpeakerContext",
    "SpeechData",
    "SpeechEvent",
    "SpeechEventType",
    "SpeechStream",
]


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: list[SpeechData] = field(default_factory=list)
    recognition_usage: RecognitionUsage | None = None
    speech_start_time: float | None = None
    """server-reported wall-clock time of speech onset, when the provider sends
    a separate speech-start signal carrying onset timing."""
    created_at: float = field(default_factory=lambda: time.time())
    """Wall-clock time when this event was created."""
    speech_end_time: float | None = None
    """Wall-clock time when the recognized speech ended, when known."""

    @classmethod
    def _from_core(cls, ev: dict[str, Any], *, offset: float = 0.0) -> SpeechEvent:
        """An event the core emits as a dict, its times shifted by ``offset``."""
        alternatives = []
        for alt in ev.pop("alternatives"):
            alt["start_time"] += offset
            alt["end_time"] += offset
            for word in alt["words"] or []:
                word.start_time = (word.start_time or 0.0) + offset
                word.end_time = (word.end_time or 0.0) + offset
                word.start_time_offset = offset
            alternatives.append(SpeechData(**alt))
        usage = ev.pop("recognition_usage")
        return cls(
            **ev,
            alternatives=alternatives,
            recognition_usage=RecognitionUsage(**usage) if usage else None,
        )


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
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._keyterms_unsupported_warned = False
        self._chat_context_unsupported_warned = False

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
        """Metadata used to label turn metrics emitted for this STT instance."""
        return {"model_name": self.model, "model_provider": self.provider}

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

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
                return await self._recognize_impl(
                    buffer, language=language, conn_options=conn_options
                )
            except APIError as e:
                if conn_options.max_retry == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                if i == conn_options.max_retry:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to recognize speech after {conn_options.max_retry + 1} attempts",
                    ) from e
                self._emit_error(e, recoverable=True)
                retry_interval = conn_options._interval_for_retry(i)
                logger.warning(
                    f"failed to recognize speech: {e}, retrying in {retry_interval}s",
                    extra={"stt": self._label, "attempt": i + 1, "streamed": False},
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
                timestamp=time.time(), label=self._label, error=api_error, recoverable=recoverable
            ),
        )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        """Set the framework-managed keyterms (session config and auto-detection), kept apart
        from the user's own; plugins supporting keyterms override it."""
        if not self._capabilities.keyterms and not self._keyterms_unsupported_warned:
            self._keyterms_unsupported_warned = True
            logger.warning(
                "keyterms are not supported by this STT, ignoring keyterms update",
                extra={"stt": self._label},
            )

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        """Feed a conversation turn to bias recognition; plugins with native context support
        set ``STTCapabilities.chat_context`` and override it."""
        if not self._capabilities.chat_context and not self._chat_context_unsupported_warned:
            self._chat_context_unsupported_warned = True
            logger.warning(
                "chat context is not supported by this STT, ignoring chat context update",
                extra={"stt": self._label},
            )

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
    """A speaker model an STT context provides, which the expressive system turns into LLM
    instructions."""

    def to_instructions(self) -> str: ...


class RecognizeStream(ABC):
    class _FlushSentinel:
        """Sentinel to mark when it was flushed"""

        pass

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ):
        """
        Args:
            stt: The STT this stream recognizes for.
            conn_options: The retry budget of the stream.
            sample_rate: The rate the audio is resampled to before recognition; the input's own
                rate when not given.
        """
        self._stt = stt
        self._conn_options = conn_options
        self._input_ch = aio.Chan[rtc.AudioFrame | RecognizeStream._FlushSentinel]()
        self._event_ch = aio.Chan[SpeechEvent]()

        self._num_retries = 0
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None

        self._start_time_offset: float = 0.0
        self._start_time: float = time.time()
        self._context: BaseModel | None = None

    @property
    def context(self) -> BaseModel | None:
        """Speaker metadata the plugin updates as it learns about the speaker."""
        return self._context

    @context.setter
    def context(self, value: BaseModel | None) -> None:
        self._context = value

    @property
    def start_time_offset(self) -> float:
        return self._start_time_offset

    @start_time_offset.setter
    def start_time_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time_offset must be non-negative")
        self._start_time_offset = value

    @property
    def start_time(self) -> float:
        """Wall-clock anchor of the stream, re-seeded on each retry; plugins may move it to when
        the first audio reached the provider, so stream-relative times convert accurately."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        if value < 0:
            raise ValueError("start_time must be non-negative")
        self._start_time = value

    def _report_connection_acquired(self, acquire_time: float, connection_reused: bool) -> None:
        """Connection timing; kept on the stream, the core measures its own."""
        self._acquire_time, self._connection_reused = acquire_time, connection_reused

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        max_retries = self._conn_options.max_retry
        # the offset accumulates across attempts, so transcripts keep linear timestamps
        last_start_time = time.time()
        while self._num_retries <= max_retries:
            try:
                self._start_time_offset += time.time() - last_start_time
                self._start_time = last_start_time = time.time()
                return await self._run()
            except APIError as e:
                # an attempt that outlived the connect timeout had connected: the budget restarts
                if time.time() - last_start_time > self._conn_options.timeout:
                    self._num_retries = 0
                if not e.retryable or max_retries == 0:
                    self._emit_error(e, recoverable=False)
                    raise
                if self._num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"failed to recognize speech after {self._num_retries} attempts",
                    ) from e
                self._emit_error(e, recoverable=True)
                retry_interval = self._conn_options._interval_for_retry(self._num_retries)
                logger.warning(
                    f"failed to recognize speech: {e}, retrying in {retry_interval}s",
                    extra={"stt": self._stt._label, "attempt": self._num_retries, "streamed": True},
                )
                await asyncio.sleep(retry_interval)
                self._num_retries += 1
            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt._emit_error(api_error, recoverable)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized"""
        self._check_input_not_ended()
        self._check_not_closed()
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")
        self._pushed_sr = frame.sample_rate
        if self._needed_sr and self._needed_sr != frame.sample_rate and not self._resampler:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._needed_sr, quality=rtc.AudioResamplerQuality.HIGH
            )
        for f in self._resampler.push(frame) if self._resampler else [frame]:
            self._input_ch.send_nowait(f)

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

    async def __anext__(self) -> SpeechEvent:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if val.type == SpeechEventType.FINAL_TRANSCRIPT:
            # a recognition succeeded: the retry budget restarts
            self._num_retries = 0
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
