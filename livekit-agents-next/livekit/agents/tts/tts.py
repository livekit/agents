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
from ..utils import aio, shortuuid

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
        # a perf_counter() stamp, as wall-clock seconds for the core
        started: float | None = self.frame.userdata.get(USERDATA_TTS_STARTED_TIME)
        return None if started is None else time.time() - (time.perf_counter() - started)

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
        # driven by plugins that manage their own segments (Rime)
        self._input_buffer: list[str | SynthesizeStream._FlushSentinel] = []
        self._mtc_text = ""
        self._mtc_pending_texts: list[str] = []
        self._input_ended = False
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


# `audio/pcm` is framed as it is; any other container is decoded (ffmpeg) as it arrives.
AudioEmitter = _core.AudioEmitter
