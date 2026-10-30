# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` and
# `#[host(record)]` declarations in:
#     rust/crates/lk-agents-abi/src/tts/tts.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, Literal, Never, get_type_hints

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, TypeVar

from livekit import rtc
from livekit.rtc import AudioFrame

from .. import _bindings, _core
from .._exceptions import APIError
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    APIConnectOptions,
)
from ..utils import aio, shortuuid
from .chat_items import MetricsMetadata
from .expressive import SpeechSteeringOptions

# The declared records.


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


TTSCapabilities.__annotations__ = get_type_hints(TTSCapabilities)
TTSCapabilities.__module__ = "livekit.agents.tts"
MarkupInfo.__annotations__ = get_type_hints(MarkupInfo)
MarkupInfo.__module__ = "livekit.agents.tts"
SynthesizedAudio.__annotations__ = get_type_hints(SynthesizedAudio)
SynthesizedAudio.__module__ = "livekit.agents.tts"


class TTSError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["tts_error"] = "tts_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class TTS(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    """A text-to-speech provider.

    Plugins implement ``synthesize()`` for one-shot synthesis, and override ``stream()`` when the
    service synthesizes text as it arrives.
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

        def llm_instructions(self, *, speech_steering: SpeechSteeringOptions | None = None) -> str | None:
            """The instructions teaching the LLM this voice's markup, ``None`` without markup."""
            return self._core().llm_instructions(speech_steering=speech_steering)

        def normalize(self, text: str) -> str:
            """Fix common LLM markup mistakes (e.g. unclosed self-closing tags)."""
            return self._core().normalize(text)

        def convert(self, text: str) -> str:
            """Lower the framework's markup to the provider's native syntax."""
            return self._core().convert(text)

    @property
    def markup(self) -> Markup:
        """Access TTS markup capabilities (instructions for LLM, text stripping)."""
        return self._markup

    @abstractmethod
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize ``text``, its attempts retried per ``conn_options``.

        Args:
            text: What to say.
            conn_options: The timeout and retries of the request.
        """

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Open a stream synthesizing text as it is pushed. Providers with the ``streaming``
        capability override it; any other is wrapped in a ``StreamAdapter``.

        Args:
            conn_options: The timeout and retries of the connection.
        """
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"  # noqa: E501
        )

    def _synthesize_with_stream(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """``synthesize()`` for a provider that only streams: one push, then the end of
        input."""
        return _ChunkedStreamFromStream(tts=self, input_text=text, conn_options=conn_options)

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

    def _core_id(self) -> dict[str, str]:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    def _core_prewarm(self) -> None:
        self.prewarm()

    def _core_sample_rate(self) -> int:
        return self.sample_rate

    def _core_num_channels(self) -> int:
        return self.num_channels

    def _core_markup_provider(self) -> str:
        return self.markup._provider_key()

    def _core_lowers_markup(self) -> bool:
        return False

    def _core_conn_options(self) -> APIConnectOptions | None:
        return None

    def _core_failed(self, error: Exception, recoverable: bool) -> None:
        self._emit_error(error, recoverable)

    async def _core_synthesize(
        self, text: str, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        out = self.synthesize(text, conn_options=conn)
        out._core_attempts.core = True
        return out

    async def _core_stream(
        self, _expressive: bool, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        out = self.stream(conn_options=conn)
        out._core_attempts.core = True
        out._expressive = _expressive
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
        self._metrics: _core.TtsMetricState | None = _core.TtsMetricState(False)
        self._core_attempts = _bindings.AttemptLoop("tts_batch", conn_options, self._emit_error)
        self._synthesize_task = asyncio.create_task(
            self._core_attempts.run(self._attempt, produced=self._produced),
            name="TTS._synthesize_task",
        )
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

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._tts._emit_error(api_error, recoverable)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None: ...

    async def _attempt(self) -> None:
        self._emitter = output_emitter = _core.AudioEmitter(
            label=self._tts.label, dst_ch=self._event_ch
        )
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if problem := self._invalid(output_emitter):
                raise APIError(problem)
        finally:
            await output_emitter.aclose()

    def _produced(self) -> bool:
        return self._emitter is not None and self._emitter.pushed_duration() > 0.0

    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(self._input_text, output_emitter.pushed_duration(), None)

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
            if self._metrics is not None:
                record = self._metrics.finish(
                    time.perf_counter(),
                    self._started_time,
                    False,
                    self._input_text,
                    _bindings.metrics_context(self._tts),
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

    Plugins implement ``_run``, reading ``input`` and pushing the provider's audio into the
    ``AudioEmitter`` it is given; its attempts are retried per ``conn_options``.
    """

    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__()
        self._tts = tts
        self._conn_options = conn_options
        self.input = aio.Chan[str | SynthesizeStream.Flush]()
        """The text ``_run`` reads, a ``Flush`` at the end of each segment; a retry reads it
        again from the start."""
        self._input_log: list[str | SynthesizeStream.Flush] = []
        self._event_ch = aio.Chan[SynthesizedAudio]()
        self._emitter: _core.AudioEmitter | None = None
        self._metrics = _core.TtsMetricState(True)
        self._segment_text = ""
        self._core_attempts = _bindings.AttemptLoop("tts_stream", conn_options, self._emit_error)
        self._task = asyncio.create_task(
            self._core_attempts.run(self._attempt, produced=self._produced, again=self._replay),
            name="TTS._main_task",
        )
        self._task.add_done_callback(lambda _: self._event_ch.close())
        self._started_time: float = 0
        self._pushed_text: str = ""
        # whether this synthesis carries expressive markup; the framework sets it at open
        self._expressive = False
        self._segment_open = False
        self._num_segments = 0

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._tts._emit_error(api_error, recoverable)

    @abstractmethod
    async def _run(self, output_emitter: _core.AudioEmitter) -> None: ...

    async def _attempt(self) -> None:
        self._emitter = output_emitter = _core.AudioEmitter(
            label=self._tts.label, dst_ch=self._event_ch
        )
        try:
            await self._run(output_emitter)
            output_emitter.end_input()
            # wait for all audio frames to be pushed & propagate errors
            await output_emitter.join()
            if problem := self._invalid(output_emitter):
                raise APIError(problem)
        finally:
            await output_emitter.aclose()

    def _produced(self) -> bool:
        return self._emitter is not None and self._emitter.pushed_duration() > 0.0

    def _replay(self) -> None:
        ended = self.input.closed
        self.input = aio.Chan[str | SynthesizeStream.Flush]()
        for item in self._input_log:
            self.input.send_nowait(item)
        if ended:
            self.input.close()

    def _invalid(self, output_emitter: _core.AudioEmitter) -> str | None:
        return _core.validate_synthesis(
            self._pushed_text,
            output_emitter.pushed_duration(),
            (self._num_segments, output_emitter.num_segments),
        )

    def mark_started(self) -> None:
        """Note that the first text went to the provider: the time its audio is measured from."""
        if self._started_time == 0:
            self._started_time = time.perf_counter()

    def push_text(self, token: str) -> None:
        """Push some text to be synthesized"""
        if not token or self.input.closed:
            return
        if not self._segment_open:
            self._segment_open = True
            self._num_segments += 1
        self._pushed_text += token
        self._segment_text += token
        self._input_log.append(token)
        self.input.send_nowait(token)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        if self.input.closed:
            return
        self._segment_open = False
        if self._segment_text:
            self._metrics.queue_text(self._segment_text)
            self._segment_text = ""
        self._input_log.append(self.Flush())
        self.input.send_nowait(self._input_log[-1])

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
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
        if self._metrics.observe(time.perf_counter(), self._started_time, val):
            record = self._metrics.finish(
                time.perf_counter(),
                self._started_time,
                False,
                "",
                _bindings.metrics_context(self._tts),
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

    def _core_push(self, token: str) -> None:
        self.push_text(token)

    def _core_flush(self) -> None:
        if not self.input.closed:
            self.flush()

    def _core_end_input(self) -> None:
        if not self.input.closed:
            self.end_input()


class _ChunkedStreamFromStream(ChunkedStream):
    """``synthesize()`` over a provider that only streams."""

    async def _run(self, output_emitter: _core.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        async with self._tts.stream(conn_options=self._conn_options) as stream:
            # The batch operation owns retries; reopening this child is one attempt.
            stream._core_attempts.core = True
            stream.push_text(self._input_text)
            stream.end_input()
            async for ev in stream:
                output_emitter.push(ev.frame.data.tobytes())
                if timed_transcripts := ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT):
                    output_emitter.push_timed_transcript(timed_transcripts)
        output_emitter.flush()


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
TTSError.__module__ = "livekit.agents.tts"
TTS.__module__ = "livekit.agents.tts"
ChunkedStream.__module__ = "livekit.agents.tts"
SynthesizeStream.__module__ = "livekit.agents.tts"
