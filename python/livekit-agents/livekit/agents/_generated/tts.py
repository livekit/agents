# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/tts/tts.py

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import cached_property
from inspect import get_annotations
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Never, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict, TypeVar

from livekit import rtc
from livekit.agents._core import SpeechSentenceTokenizer
from livekit.agents.tokenize import SentenceTokenizer

from .. import _bindings, _core, tokenize
from .._bindings import APIError
from ..tokenize import SentenceTokenizer as HostSentenceTokenizer
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio, is_given, shortuuid
from .chat_items import MetricsMetadata
from .expressive import SpeechSteeringOptions
from .rtc_frames import AudioFrame
from .stream_pacer import SentenceStreamPacer
from .types import ProviderId
from .usage import TokenUsage

if TYPE_CHECKING:
    from builtins import object as _HostOptionsTokenizer


# Record declarations:
#     rust/crates/lk-agents-abi/src/tokenize/bindings.rs
#     rust/crates/lk-agents-abi/src/tts/markup.rs
#     rust/crates/lk-agents-abi/src/tts/streams.rs
#     rust/crates/lk-agents-abi/src/tts/tts.rs


SentenceTokenizerLeg: TypeAlias = SpeechSentenceTokenizer | SentenceTokenizer
"""What splits a TTS's text: a built-in `tokenize.SentenceTokenizer`, which the stage runs
natively with its settings, or a custom one, whose sentences it reads from the host.
"""


class ExpressiveTag(TypedDict):
    """An expressive markup tag stripped from a transcript: the tag name (or the expr marker's type)
    and its payload (the ``value`` attribute, the expr ``label`` or the inner text).
    """

    type: str
    """The markup tag name (``"emotion"``, ``"sound"``, ...) or the expr marker type."""
    value: str
    """The payload: the ``value="..."`` attribute, the expr ``label``, or the tag's inner text."""


@dataclass
class MarkupSplit:
    """Text with its expressive markup stripped, and the tags taken out of it."""

    text: str
    tags: list[ExpressiveTag]


@dataclass
class SegmentCount:
    """A streamed synthesis's segments: the flushes asked for (`expected`) and the segments pushed."""

    expected: int
    actual: int


@dataclass
class TTSCapabilities:
    """What a TTS supports, fixed at construction.

    Usage: a figure the provider reports always wins. The characters it reports it billed
    (`report_usage`) replace the ones the framework counts for that segment.
    """

    streaming: bool
    """Whether this TTS supports streaming (generally using websockets)"""
    aligned_transcript: bool = False
    """Whether this TTS supports aligned transcripts with word timestamps"""
    text_unit: Literal["sentences", "words", "text"] = "sentences"
    """What each chunk of a stream's input holds: a provider that buffers text itself takes words"""
    max_text_len: int | None = None
    """The most characters one request may carry; None takes the markup's known limit"""


@dataclass
class MarkupInfo:
    """What the expressive markup pipeline can do with a given voice."""

    nonverbals: dict[Literal["laughing", "breathing", "sighing", "crying", "vocalizing", "mouth_sounds", "reflex_sounds"], list[str]] = field(default_factory=dict)
    """``NonverbalOptions`` field -> the labels it governs; an absent field is a no-op"""


@dataclass
class SynthesizedAudio:
    frame: AudioFrame
    """Synthesized audio frame"""
    request_id: str
    """Request ID (one segment could be made up of multiple requests)"""
    is_final: bool = False
    """Whether this is latest frame of the segment"""
    segment_id: str = ""
    """Segment ID, each segment is separated by a flush (streaming only)"""
    delta_text: str = ""
    """Current segment of the synthesized audio (streaming only)"""
    usage: TokenUsage | None = None
    """Token usage the provider reported for the segment, on its final frame"""
    characters_count: int | None = None
    """Characters the provider billed for the segment, on its final frame"""
    provider_request_ids: list[str] = field(default_factory=list)
    """The provider's request ids learnt since the previous frame, in the order learnt"""


@dataclass
class TTSOptions:
    """`tts.TTSOptions`: how the framework prepares a TTS's text."""

    sentence_tokenizer: HostSentenceTokenizer = field(default_factory=tokenize.SentenceTokenizer)
    """Splits the text into the chunks the TTS is sent: the built-in ``SentenceTokenizer`` by
    default, with the TTS's own tags kept whole and its request limit applied on top. A TTS
    that takes words is sent words whatever this is.
    """
    pacing: SentenceStreamPacer | None = None
    """A ``SentenceStreamPacer`` holds later chunks until the audio plays out, for more context
    per request; ``None`` sends each as soon as it is split. A TTS without streaming is paced
    by default, a streaming one is not.
    """


OptionsTokenizer: TypeAlias = "_HostOptionsTokenizer"


@dataclass
class SpeechChunk:
    """One chunk of a TTS's input, as the stage prepared it."""

    text: str
    """What the provider is sent: its markup in the provider's syntax."""
    said: str
    """The words it says, without markup: what its audio is aligned to."""


class TTSError(BaseModel):
    """The public provider error event; its Python exception is retained but never serialized."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["tts_error"] = "tts_error"
    timestamp: float
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool


MarkupSplit.__annotations__ = get_annotations(MarkupSplit, eval_str=True)
MarkupSplit.__module__ = "livekit.agents.tts"
TTSCapabilities.__annotations__ = get_annotations(TTSCapabilities, eval_str=True)
TTSCapabilities.__module__ = "livekit.agents.tts"
MarkupInfo.__annotations__ = get_annotations(MarkupInfo, eval_str=True)
MarkupInfo.__module__ = "livekit.agents.tts"
SynthesizedAudio.__annotations__ = get_annotations(SynthesizedAudio, eval_str=True)
SynthesizedAudio.__module__ = "livekit.agents.tts"
TTSOptions.__annotations__ = get_annotations(TTSOptions, eval_str=True)
TTSOptions.__module__ = "livekit.agents.tts"
SpeechChunk.__annotations__ = get_annotations(SpeechChunk, eval_str=True)
SpeechChunk.__module__ = "livekit.agents.tts"
TTSError.__module__ = "livekit.agents.tts"


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class TTS(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    """A text-to-speech provider.

    ``synthesize()`` and ``stream()`` are the framework's: the text goes through its stage (split,
    paced, lowered to the provider's markup) and each attempt is retried. Plugins implement one
    provider attempt: ``_synthesize()``, and ``_stream()`` when the service synthesizes text as it
    arrives.
    """

    def __init__(
        self, *, capabilities: TTSCapabilities, sample_rate: int, num_channels: int
    ) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._markup = self.Markup(self)
        self._options_cell = _core.TtsOptionsCell(tokenize.SentenceTokenizer(), capabilities)

    @property
    def label(self) -> str:
        """The name logs and metrics give this instance: its class path."""
        return self._label

    @property
    def capabilities(self) -> TTSCapabilities:
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

    @property
    def sample_rate(self) -> int:
        """The rate of the audio the provider produces, which the framework resamples from."""
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """The channels of the audio the provider produces."""
        return self._num_channels

    class Markup:
        """The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to
        write and how it is normalized and lowered before synthesis.

        Plugins opt in by overriding ``_provider_key``.
        """

        def __init__(self, tts: TTS) -> None:
            self._tts = tts

        def _provider_key(self) -> str:
            """The provider's key into the markup tables, or ``""`` for no markup."""
            return ""

        def _core(self) -> _core.Markup:
            return _core.Markup(self._provider_key())

        @property
        def info(self) -> MarkupInfo:
            """The queryable markup matrix for this voice."""
            return _core.supported_nonverbals(self._provider_key())

        def llm_instructions(
            self, *, speech_steering: SpeechSteeringOptions | None = None
        ) -> str | None:
            """The instructions teaching the LLM this voice's markup, ``None`` without markup."""
            return self._core().llm_instructions(speech_steering=speech_steering)

    @property
    def markup(self) -> Markup:
        """Access TTS markup capabilities (instructions for LLM, text stripping)."""
        return self._markup

    @property
    def options(self) -> TTSOptions:
        """How the framework prepares this TTS's text: its sentence tokenizer and pacing, changed
        with ``update_options``."""
        cell = self._core_options()
        tokenizer = cast(tokenize.SentenceTokenizer, cell.sentence_tokenizer)
        return TTSOptions(sentence_tokenizer=tokenizer, pacing=cell.pacing)

    def update_options(
        self,
        *,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Change how the framework prepares this TTS's text, from its next synthesis on, wherever
        the TTS was handed: a session, a ``StreamAdapter`` or a ``FallbackAdapter``.

        Args:
            sentence_tokenizer: What splits the text into the chunks the TTS is sent.
            pacing: A ``SentenceStreamPacer``, or ``None`` to send each chunk as soon as it is
                split.
        """
        cell = self._core_options()
        tokenizer = (
            sentence_tokenizer
            if is_given(sentence_tokenizer)
            else cast(tokenize.SentenceTokenizer, cell.sentence_tokenizer)
        )
        paced = pacing if is_given(pacing) else cell.pacing
        cell.set(tokenizer, paced)

    def synthesize(
        self, text: str, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> ChunkedStream:
        """Synthesize ``text``, its attempts retried per ``conn_options``.

        Args:
            text: What to say.
            conn_options: The timeout and retries of the request; the TTS's own by default.
        """
        return _Synthesis(tts=self, input_text=text, conn_options=self._conn(conn_options))

    def stream(
        self, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> SynthesizeStream:
        """Open a stream synthesizing text as it is pushed. A TTS without the ``streaming``
        capability needs a ``StreamAdapter``.

        Args:
            conn_options: The timeout and retries of the connection; the TTS's own by default.
        """
        return _Synthesizing(tts=self, conn_options=self._conn(conn_options))

    def _conn(self, given: NotGivenOr[APIConnectOptions]) -> APIConnectOptions:
        if is_given(given):
            return given
        return self._core_conn_options() or DEFAULT_API_CONNECT_OPTIONS

    def _synthesize(self, chunk: SpeechChunk, *, conn_options: APIConnectOptions) -> ChunkedStream:
        """One provider attempt at a whole text: ``chunk.text`` in the provider's syntax,
        ``chunk.said`` the words it says. The framework retries it. Every plugin implements it;
        a TTS the core runs natively has none."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _synthesize")

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        """One provider attempt at a stream: its ``input`` yields each ``SpeechChunk`` the
        framework prepared, a ``Flush`` at each segment's end. Providers with the ``streaming``
        capability override it."""
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"  # noqa: E501
        )

    def _synthesize_with_stream(
        self, chunk: SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        """``_synthesize()`` for a provider that only streams: one chunk, then the end of
        input."""
        return _ChunkedStreamFromStream(tts=self, chunk=chunk, conn_options=conn_options)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            TTSError(
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

    def _core_capabilities(self) -> TTSCapabilities:
        return self.capabilities

    def _core_id(self) -> ProviderId:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    def _core_prewarm(self) -> None:
        self.prewarm()

    def _core_sample_rate(self) -> int:
        return self.sample_rate

    def _core_num_channels(self) -> int:
        return self.num_channels

    def _core_markup_provider(self) -> str:
        return self.markup._provider_key()

    def _core_retries_itself(self) -> bool:
        return False

    def _core_options(self) -> _core.TtsOptionsCell:
        return self._options_cell

    def _core_conn_options(self) -> APIConnectOptions | None:
        return None

    def _core_failed(self, error: Exception, recoverable: bool) -> None:
        self._emit_error(error, recoverable)

    async def _core_synthesize(
        self, chunk: SpeechChunk, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        out = self._synthesize(chunk, conn_options=conn)
        out._core_attempts.core = True
        return out

    async def _core_stream(
        self, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        out = self._stream(conn_options=conn)
        out._core_attempts.core = True
        return out


class ChunkedStream(ABC):
    """One synthesis of a whole text: ``SynthesizedAudio`` out.

    Plugins implement ``_run``, pushing the provider's audio into the ``AudioEmitter`` it is given;
    its attempts are retried per ``conn_options``.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        self._input_text = input_text
        self._tts = tts
        self._conn_options = conn_options
        # the full text is submitted to the provider at creation time
        self._started_time: float = time.perf_counter()
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._emitter: _core.AudioEmitter | None = None
        self._metrics: _core.TtsMetricState | None = _core.TtsMetricState(
            False, tts.label, tts.model, tts.provider
        )
        self._core_attempt_task: asyncio.Task[None] | None = None
        self._core_cancelled = False
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

    @cached_property
    def _core_attempts(self) -> _core.ProviderAttempts:
        """The attempts ``_run`` is retried within."""
        return _core.ProviderAttempts("tts_batch", self._conn_options, self)

    def _core_retrying(self, error: Exception) -> None:
        self._tts._emit_error(error, True)

    def _core_failed(self, error: Exception) -> None:
        self._tts._emit_error(error, False)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        """One attempt: synthesize ``input_text`` into ``output_emitter``.

        Called once per attempt with a fresh emitter; a retry calls it again from the start.
        Raise ``APIError`` (its ``kind`` decides the retry) to fail the attempt;
        once audio was pushed a failure is not retried. Cancellation means a deadline passed
        or the stream was closed: clean up and let it propagate.
        """

    async def _main_task(self) -> None:
        # run() starts from the task, after the framework set ``core``
        await _bindings.run_attempts(self, self._core_attempts)

    async def _core_attempt(self, retry: bool) -> None:
        _bindings.attempt_started(self)
        await self._core_run(self._invalid)

    async def _core_run(
        self, invalid: Callable[[_core.AudioEmitter], str | None] | None = None
    ) -> None:
        """``_run`` once, into a fresh emitter whose audio reaches the reader even when it fails."""
        self._emitter = output_emitter = _core.AudioEmitter(
            label=self._tts.label, dst_ch=self._event_ch
        )
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if invalid is not None and (problem := invalid(output_emitter)):
                raise APIError(problem, kind="protocol")
        except Exception:
            await _deliver_pushed(output_emitter)
            raise
        finally:
            await output_emitter.aclose()

    def _core_produced(self) -> bool:
        return self._emitter is not None and self._emitter.pushed_duration() > 0.0

    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(self._input_text, output_emitter.pushed_duration(), None)

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        return rtc.combine_audio_frames([ev.frame async for ev in self])

    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        await aio.cancel_and_wait(
            self._synthesize_task, timeout=self._core_attempts.cleanup_timeout
        )
        self._event_ch.close()
        # The reader is done with the stream: what was buffered for it is dropped, not returned.
        while not self._event_ch.empty():
            self._event_ch.recv_nowait()

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._synthesize_task.cancelled() and (exc := self._synthesize_task.exception()):
                raise exc  # noqa: B904
            if self._metrics is not None:
                record = self._metrics.finish(
                    time.perf_counter(),
                    self._started_time,
                    False,
                    self._input_text,
                    time.time(),
                )
                self._metrics = None
                _bindings.emit_metrics(self._tts, record)
            raise StopAsyncIteration from None
        if self._metrics is not None:
            self._metrics.observe(time.perf_counter(), self._started_time, val)
        val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SynthesizeStream(ABC):
    """A live synthesis: text in through ``push_text``, ``SynthesizedAudio`` out.

    Plugins implement ``_run`` for one provider attempt, reading ``input`` (each ``SpeechChunk``
    the framework prepared, a ``Flush`` at each segment's end) and pushing the provider's audio
    into the ``AudioEmitter`` it is given.
    """

    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self.input = aio.Chan[SpeechChunk | SynthesizeStream.Flush]()
        """The chunks ``_run`` reads, a ``Flush`` at the end of each segment; a retry reads them
        again from the start."""
        self._input_log: list[SpeechChunk | SynthesizeStream.Flush] = []
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._emitter: _core.AudioEmitter | None = None
        self._metrics: _core.TtsMetricState | None = _core.TtsMetricState(
            True, tts.label, tts.model, tts.provider
        )
        self._segment_text = ""
        self._core_attempt_task: asyncio.Task[None] | None = None
        self._core_cancelled = False
        self._task = asyncio.create_task(self._main_task(), name="TTS._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._started_time: float = 0
        self._pushed_text: str = ""
        self._segment_open = False
        self._num_segments = 0

    @cached_property
    def _core_attempts(self) -> _core.ProviderAttempts:
        """The attempts ``_run`` is retried within."""
        return _core.ProviderAttempts("tts_stream", self._conn_options, self)

    def _core_retrying(self, error: Exception) -> None:
        self._tts._emit_error(error, True)

    def _core_failed(self, error: Exception) -> None:
        self._tts._emit_error(error, False)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        """One attempt: read ``input`` and synthesize it into ``output_emitter``.

        Called once per attempt with a fresh emitter; a retry calls it again with ``input``
        replayed from the start. Raise ``APIError`` (its ``kind`` names what failed and
        decides the retry) to fail the attempt; once audio was pushed a failure is not retried.
        Cancellation means a deadline passed or the stream was closed: clean up and let it
        propagate.
        """

    async def _main_task(self) -> None:
        # run() starts from the task, after the framework set ``core``
        await _bindings.run_attempts(self, self._core_attempts)

    async def _core_attempt(self, retry: bool) -> None:
        _bindings.attempt_started(self)
        if retry:
            self._replay()
        await self._core_run(self._invalid)

    async def _core_run(
        self, invalid: Callable[[_core.AudioEmitter], str | None] | None = None
    ) -> None:
        """``_run`` once, into a fresh emitter whose audio reaches the reader even when it fails."""
        self._emitter = output_emitter = _core.AudioEmitter(
            label=self._tts.label, dst_ch=self._event_ch
        )
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if invalid is not None and (problem := invalid(output_emitter)):
                raise APIError(problem, kind="protocol")
        except Exception:
            await _deliver_pushed(output_emitter)
            raise
        finally:
            await output_emitter.aclose()

    def _core_produced(self) -> bool:
        return self._emitter is not None and self._emitter.pushed_duration() > 0.0

    def _replay(self) -> None:
        ended = self.input.closed
        self.input = aio.Chan[SpeechChunk | SynthesizeStream.Flush]()
        for item in self._input_log:
            self.input.send_nowait(item)
        if ended:
            self.input.close()

    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(
            self._pushed_text,
            output_emitter.pushed_duration(),
            SegmentCount(expected=self._num_segments, actual=output_emitter.num_segments),
        )

    def mark_started(self) -> None:
        """Note that the first text went to the provider: the time its audio is measured from."""
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        raise RuntimeError("push text to the stream tts.stream() returns")

    def _push_chunk(self, chunk: SpeechChunk) -> None:
        if not chunk.text or self.input.closed:
            return
        if not self._segment_open:
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += chunk.said
        self._segment_text += chunk.text
        self._input_log.append(chunk)
        self.input.send_nowait(chunk)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self.input.closed:
            return
        self._segment_open = False
        if self._segment_text and self._metrics is not None:
            self._metrics.queue_text(self._segment_text)
            self._segment_text = ""
        self._input_log.append(self.Flush())
        self.input.send_nowait(self._input_log[-1])

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately"""
        await aio.cancel_and_wait(self._task, timeout=self._core_attempts.cleanup_timeout)
        self._event_ch.close()
        # The reader is done with the stream: what was buffered for it is dropped, not returned.
        while not self._event_ch.empty():
            self._event_ch.recv_nowait()
        self.input.close()

    async def __anext__(self) -> SynthesizedAudio:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if self._started_time:
            val.frame.userdata[USERDATA_TTS_STARTED_TIME] = self._started_time
        # a segment's last frame closes its metrics; the next segment is timed from its own start
        if self._metrics is not None and self._metrics.observe(
            time.perf_counter(), self._started_time, val
        ):
            record = self._metrics.finish(
                time.perf_counter(),
                self._started_time,
                False,
                "",
                time.time(),
            )
            _bindings.emit_metrics(self._tts, record)
            self._started_time = 0
        return val

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _core_push(self, chunk: SpeechChunk) -> None:
        self._push_chunk(chunk)

    def _core_flush(self) -> None:
        if not self.input.closed:
            self.flush()

    def _core_end_input(self) -> None:
        if not self.input.closed:
            self.end_input()


class _Synthesis(ChunkedStream):
    """``tts.synthesize()``: a view of the framework's native request over the TTS, which prepares
    the text, retries the provider's attempts and reports each failure; this only reads its audio.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        self._native = _core.TtsStream(tts=tts, text=input_text, conn_options=conn_options)
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        # a host plugin's own attempts report the metrics; a native provider has none
        if not self._native.native:
            self._metrics = None

    async def _main_task(self) -> None:
        await self._core_run()

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        _initialize(output_emitter, self._tts, stream=False)
        try:
            await self._native.drain_into(output_emitter, stream=False)
        finally:
            await self._native.aclose()


class _Synthesizing(SynthesizeStream):
    """``tts.stream()``: a view of the framework's native request over the TTS; what is pushed
    goes to its stage, which prepares each chunk for the provider. The request retries the
    provider's attempts and reports each failure; this only reads its audio."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        self._native = _core.TtsStream(tts=tts, conn_options=conn_options)
        super().__init__(tts=tts, conn_options=conn_options)
        # a host plugin's own attempts report the metrics; a native provider has none
        if not self._native.native:
            self._metrics = None

    def _check_input(self) -> None:
        """Raise the synthesis's failure, or that the stream is closed or its input ended."""
        _bindings.check_input(
            self, self._task, closed=self._event_ch.closed, ended=self.input.closed
        )

    def push_text(self, token: str) -> None:
        self._check_input()
        if not token:
            return
        if not self._segment_open:
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += token
        self._segment_text += token
        self.mark_started()
        self._native.push_text(token)

    def flush(self) -> None:
        self._check_input()
        self._segment_open = False
        if self._segment_text and self._metrics is not None:
            self._metrics.queue_text(self._segment_text)
            self._segment_text = ""
        self._native.flush()

    def end_input(self) -> None:
        self.flush()
        self.input.close()
        self._native.end_input()

    async def _main_task(self) -> None:
        await self._core_run()

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        _initialize(output_emitter, self._tts, stream=True)
        try:
            await self._native.drain_into(output_emitter, stream=True)
        finally:
            await self._native.aclose()


class _ChunkedStreamFromStream(ChunkedStream):
    """``_synthesize()`` over a provider that only streams: its stream fed the one chunk."""

    def __init__(self, *, tts: TTS, chunk: SpeechChunk, conn_options: APIConnectOptions) -> None:
        self._chunk = chunk
        super().__init__(tts=tts, input_text=chunk.text, conn_options=conn_options)

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        async with self._tts._stream(conn_options=self._conn_options) as stream:
            # The batch operation owns retries; reopening this child is one attempt.
            stream._core_attempts.core = True
            stream._push_chunk(self._chunk)
            stream.end_input()
            async for ev in stream:
                output_emitter.push(ev.frame.data.tobytes())
                if timed_transcripts := ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(timed_transcripts)
        output_emitter.flush()


async def _deliver_pushed(output_emitter: _core.AudioEmitter) -> None:
    """A failed attempt still delivers the audio it pushed, before its error is raised."""
    try:
        output_emitter.end_input()
        await output_emitter.join()
    except RuntimeError:
        # never initialized, already ended, or a decode failure: the attempt's error stands
        pass


def _initialize(output_emitter: _core.AudioEmitter, tts: TTS[Any], *, stream: bool) -> None:
    output_emitter.initialize(
        request_id=shortuuid(),
        sample_rate=tts.sample_rate,
        num_channels=tts.num_channels,
        mime_type="audio/pcm",
        stream=stream,
    )




# The public module identity is assigned after all classes are built.
TTS.__module__ = "livekit.agents.tts"
ChunkedStream.__module__ = "livekit.agents.tts"
SynthesizeStream.__module__ = "livekit.agents.tts"
