# Plugin base classes, hand-written. lk-agents-codegen writes them into `_generated/stt.py` after
# the module's records and checks they answer every call the proxies in `stt.rs` declare.

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Generic, Literal, Never

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, TypeVar

from livekit import rtc

from .. import _bindings, _core
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import AudioBuffer, aio, is_given
from .chat_items import MetricsMetadata
from .stt import SpeechEvent, STTCapabilities


class STTError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


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

    def _core_id(self) -> dict[str, str]:
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

    Plugins implement ``_run``, reading ``input`` and handing each event to ``send``.
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
        self._event_ch = aio.Chan[SpeechEvent](maxsize=64)
        self._start_time = time.time()
        self._start_time_offset = 0.0
        self._context: BaseModel | None = None
        self._needed_sr = sample_rate if is_given(sample_rate) else None
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None
        self._core_attempts = _bindings.AttemptLoop("stt_stream", conn_options, self._emit_error)
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

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
    async def _run(self) -> None: ...

    def _input_received(self, item: rtc.AudioFrame | RecognizeStream.Flush) -> None:
        if isinstance(item, self.Flush):
            self._core_attempts.expect_response()

    def _input_ended(self) -> None:
        self._core_attempts.input_ended()

    async def _main_task(self) -> None:
        started = time.time()

        # each attempt moves the timeline on by the time the last one ran, so a reconnected
        # stream's timestamps stay linear
        async def attempt() -> None:
            nonlocal started
            now = time.time()
            self._start_time_offset += now - started
            self._start_time = started = now
            await self._run()

        await self._core_attempts.run(attempt)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._stt._emit_error(api_error, recoverable)

    def _send_input(self, item: rtc.AudioFrame | RecognizeStream.Flush) -> None:
        try:
            self.input.send_nowait(item)
        except aio.ChanFull:
            self._core_attempts.capacity_exceeded(input=True)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized."""
        self._check_not_closed()
        self._check_input_not_ended()
        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")
        self._pushed_sr = frame.sample_rate
        if self._needed_sr and self._needed_sr != frame.sample_rate and not self._resampler:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._needed_sr, quality=rtc.AudioResamplerQuality.HIGH
            )
        for f in self._resampler.push(frame) if self._resampler else [frame]:
            self._send_input(f)

    def flush(self) -> None:
        """Mark the end of the current segment."""
        self._check_not_closed()
        self._check_input_not_ended()
        if self._resampler:
            for frame in self._resampler.flush():
                self._send_input(frame)
        self._send_input(self.Flush())

    def end_input(self) -> None:
        """Mark the end of input: no more audio will be pushed."""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately."""
        self.input.close()
        await _bindings.cancel_and_wait_bounded(
            self._task, timeout=self._core_attempts.cleanup_timeout
        )
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
        record, _ = _core.stt_stream_metrics(event, _bindings.metrics_context(self._stt))
        _bindings.emit_metrics(self._stt, record)
        try:
            self._event_ch.send_nowait(event)
        except aio.ChanFull:
            self._core_attempts.capacity_exceeded(input=False)

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

    def _core_push(self, frame: rtc.AudioFrame) -> None:
        self.push_frame(frame)

    def _core_flush(self) -> None:
        if not self.input.closed:
            self.flush()

    def _core_end_input(self) -> None:
        if not self.input.closed:
            self.end_input()
