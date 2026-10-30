# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` and
# `#[host(record)]` declarations in:
#     rust/crates/lk-agents-abi/src/vad/vad.py
#     rust/crates/lk-agents-abi/src/vad/events.rs
#     rust/crates/lk-agents-abi/src/vad/vad.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum, unique
from types import TracebackType
from typing import Generic, Literal, Never, Self, get_type_hints

from typing_extensions import TypeVar

from livekit import rtc
from livekit.rtc import AudioFrame

from .._exceptions import APIError
from ..utils import aio
from .chat_items import MetricsMetadata

# The declared records.


@unique
class VADEventType(StrEnum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    """
    Represents an event detected by the Voice Activity Detector (VAD).
    """

    type: VADEventType
    """Type of the VAD event (e.g., start of speech, end of speech, inference done)."""
    samples_index: int
    """Index of the audio sample where the event occurred, relative to the inference sample rate."""
    timestamp: float
    """Timestamp (in seconds) when the event was fired."""
    speech_duration: float
    """Duration of the speech segment in seconds."""
    silence_duration: float
    """Duration of the silence segment in seconds."""
    frames: list[AudioFrame] = field(default_factory=list)
    """
    List of audio frames associated with the speech.

    - For `start_of_speech` events, this contains the audio chunks that triggered the detection.
    - For `inference_done` events, this contains the audio chunks that were processed.
    - For `end_of_speech` events, this contains the complete user speech.
    """
    probability: float = 0.0
    """Probability that speech is present (only for `INFERENCE_DONE` events)."""
    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events)."""
    speaking: bool = False
    """Indicates whether speech was detected in the frames."""
    raw_accumulated_silence: float = 0.0
    """Threshold used to detect silence."""
    raw_accumulated_speech: float = 0.0
    """Threshold used to detect speech."""


@dataclass
class VADCapabilities:
    update_interval: float


VADEventType.__module__ = "livekit.agents"
VADEvent.__annotations__ = get_type_hints(VADEvent)
VADEvent.__module__ = "livekit.agents"
VADCapabilities.__annotations__ = get_type_hints(VADCapabilities)
VADCapabilities.__module__ = "livekit.agents"


TEvent = TypeVar("TEvent", default=Never, contravariant=True)

# The native VAD stream's bounds (lk-agents-inference vad/stream.rs): 2.56 s of 10 ms frames, at
# most 2 MiB of pending audio, one second of 48 kHz stereo per frame and 64 pending events.
_INPUT_CAPACITY = 256
_INPUT_BYTES = 2 * 1024 * 1024
_MAX_FRAME_BYTES = 192_000
_OUTPUT_CAPACITY = 64


class _Input(aio.Chan["rtc.AudioFrame | VADStream.Flush"]):
    """A stream's audio, read only while its reader keeps up with the events, as the native
    stream's worker waits for event space: pending events stop the detector, and the bounded
    input then refuses more audio."""

    def __init__(self, stream: VADStream) -> None:
        super().__init__(maxsize=_INPUT_CAPACITY, _on_receive=stream._input_received)
        self._events_drained = stream._events_drained

    async def recv(self) -> rtc.AudioFrame | VADStream.Flush:
        await self._events_drained.wait()
        return await super().recv()


class VAD(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected"] | TEvent],
    Generic[TEvent],
):
    """A voice activity detector.

    Plugins implement ``stream()``, returning a ``VADStream`` whose ``_main_task`` turns the pushed
    audio into ``VADEvent``s.
    """

    def __init__(self, *, capabilities: VADCapabilities) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._capabilities = capabilities

    @property
    def label(self) -> str:
        """The name logs and metrics give this instance: its class path."""
        return self._label

    @property
    def capabilities(self) -> VADCapabilities:
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
        """The model and provider labelling this instance's metrics."""
        return {"model_name": self.model, "model_provider": self.provider}

    def prewarm(self) -> None:
        """Load the model early; a detector with nothing to warm ignores it."""

    @abstractmethod
    def stream(self) -> VADStream:
        """Open a stream detecting speech in the audio pushed into it."""

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

    def _core_label(self) -> str:
        return self.label

    def _core_capabilities(self) -> VADCapabilities:
        return self.capabilities

    def _core_id(self) -> dict[str, str]:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    def _core_prewarm(self) -> None:
        self.prewarm()

    async def _core_stream(self) -> VADStream:
        return self.stream()


class VADStream(ABC):
    """A live detection: audio in through ``push_frame``, ``VADEvent`` out.

    Plugins implement ``_main_task``, reading ``input`` and handing each event to ``send``.
    """

    class Flush:
        """Marks the end of a segment in ``input``."""

    def __init__(self, vad: VAD) -> None:
        self._vad = vad
        self._input_bytes = 0
        self._events_drained = asyncio.Event()
        self._events_drained.set()
        self.input: aio.Chan[rtc.AudioFrame | VADStream.Flush] = _Input(self)
        """The audio ``_main_task`` reads, a ``Flush`` at the end of each segment."""
        self._event_ch = aio.Chan[VADEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

    def _input_received(self, item: rtc.AudioFrame | VADStream.Flush) -> None:
        if isinstance(item, rtc.AudioFrame):
            self._input_bytes -= item.data.nbytes

    def _send_input(self, item: rtc.AudioFrame | VADStream.Flush) -> None:
        try:
            self.input.send_nowait(item)
        except aio.ChanFull:
            raise APIError("VAD input capacity exhausted", retryable=False) from None

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push some audio frame to be analyzed. Audio beyond the stream's bounds raises
        ``APIError`` and is not kept."""
        self._check_input_not_ended()
        self._check_not_closed()
        size = frame.data.nbytes
        if size > _MAX_FRAME_BYTES:
            raise APIError(
                f"VAD input frame exceeds the {_MAX_FRAME_BYTES}-byte limit", retryable=False
            )
        if self._input_bytes + size > _INPUT_BYTES:
            raise APIError("VAD input byte capacity exhausted", retryable=False)
        self._send_input(frame)
        self._input_bytes += size

    def flush(self) -> None:
        """Mark the end of the current segment.

        Implementations MUST treat this as a hard segment boundary: drop any accumulated
        speech/silence state so the next pushed frame starts a fresh segment. Used by the
        pipeline to recover from out-of-band end-of-turn signals (e.g. STT EOS) without
        tearing down and recreating the stream.
        """
        self._check_input_not_ended()
        self._check_not_closed()
        self._send_input(self.Flush())

    def end_input(self) -> None:
        """Mark the end of input, no more audio will be pushed"""
        self.flush()
        self.input.close()

    async def aclose(self) -> None:
        """Close the stream immediately"""
        self.input.close()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> VADEvent:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904
            raise StopAsyncIteration from None
        if self._event_ch.qsize() < _OUTPUT_CAPACITY:
            self._events_drained.set()
        return val

    def __aiter__(self) -> AsyncIterator[VADEvent]:
        return self

    def send(self, event: VADEvent) -> None:
        """Hand ``event`` to the reader. Once it falls behind, ``input`` waits for it."""
        self._event_ch.send_nowait(event)
        if self._event_ch.qsize() >= _OUTPUT_CAPACITY:
            self._events_drained.clear()

    @property
    def closed(self) -> bool:
        """Whether the stream is closed: its reader takes nothing more."""
        return self._event_ch.closed

    def _check_not_closed(self) -> None:
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


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
VAD.__module__ = "livekit.agents"
VADStream.__module__ = "livekit.agents"
