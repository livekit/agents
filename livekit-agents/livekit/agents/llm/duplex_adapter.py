from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr, TimedString
from ..utils import aio, shortuuid
from .chat_context import ChatContext, FunctionCall
from .duplex import (
    DuplexAudioFrame,
    DuplexCapabilities,
    DuplexModel,
    DuplexSession,
    DuplexTranscriptDelta,
    DuplexTurnEndedEvent,
    DuplexTurnStartedEvent,
)
from .realtime import (
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeError,
    RealtimeModel,
    RealtimeSession,
)
from .tool_context import Tool, ToolChoice, ToolContext

# a floor this low is digital silence; it keeps the gate's ratios finite when a model emits zeros
_SILENCE_FLOOR = 1e-4

# liveness bound only: a turn closes on its transcript catching up, or on the model ending it
_STALLED_TRANSCRIPT_TIMEOUT = 3.0


def _frame_rms(frame: rtc.AudioFrame) -> float:
    """Root-mean-square level of a frame, normalized to 0..1."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples)))) / 32768.0


class AudioGate(Protocol):
    """Decides which frames of a continuously-emitting model carry output worth playing."""

    @property
    def voiced(self) -> bool:
        """Whether the last frame carried sound, regardless of any forcing."""
        ...

    def update(self, frame: rtc.AudioFrame, *, forced: bool = False) -> bool:
        """True while the frame belongs to an open burst of output.

        ``forced`` marks audio the model attributes to a turn: always output, and it holds the gate.
        """
        ...


class AdaptiveNoiseGate:
    """Opens on output that stands out from the model's own noise floor.

    Thresholds are ratios against the quietest frame of the recent past, which speech cannot drag
    upward, and every duration is measured in audio rather than wall clock. One set of defaults
    therefore ports across providers, frame sizes and network conditions.
    """

    def __init__(
        self,
        *,
        open_ratio: float = 3.0,
        close_ratio: float = 1.8,
        # rides out the ~400 ms pauses inside an utterance without merging it into the next
        hangover: float = 0.5,
        window: float = 10.0,
    ) -> None:
        self._open_ratio = open_ratio
        self._close_ratio = close_ratio
        self._hangover = hangover
        self._window = window
        self._history: deque[tuple[float, float]] = deque()
        self._history_duration = 0.0
        self._open = False
        self._quiet = 0.0
        self._voiced = False

    @property
    def voiced(self) -> bool:
        return self._voiced

    def update(self, frame: rtc.AudioFrame, *, forced: bool = False) -> bool:
        rms = _frame_rms(frame)

        self._history.append((rms, frame.duration))
        self._history_duration += frame.duration
        while self._history_duration > self._window and len(self._history) > 1:
            self._history_duration -= self._history.popleft()[1]
        floor = max(min(level for level, _ in self._history), _SILENCE_FLOOR)
        self._voiced = rms > floor * self._close_ratio

        if forced:
            self._open = True
            self._quiet = 0.0
            return True

        if not self._open:
            if rms > floor * self._open_ratio:
                self._open = True
                self._quiet = 0.0
        elif rms < floor * self._close_ratio:
            self._quiet += frame.duration
            if self._quiet >= self._hangover:
                self._open = False
        else:
            self._quiet = 0.0

        return self._open


@dataclass
class _Burst:
    """One contiguous stretch of model output, presented to the framework as a generation."""

    id: str
    message_ch: aio.Chan[MessageGeneration]
    function_ch: aio.Chan[FunctionCall]
    text_ch: aio.Chan[str]
    audio_ch: aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    model_message_id: str | None = None
    transcript: str = ""
    has_audio: bool = False
    transcript_end_ms: int | None = None
    orphaned: bool = False
    """Opened by a fragment that outlived its audio, so no turn can claim it."""
    turn_ended: bool = False
    """The model has declared this burst's turn over."""

    # anchors mapping the model's timeline onto playback; audio the model omits never plays
    _start_ms: int | None = field(default=None)
    _skipped_ms: int = field(default=0)
    _end_ms: int | None = field(default=None)
    _voiced_end_ms: int | None = field(default=None)
    _last_annotation: float = field(default=0.0)

    @property
    def transcript_pending(self) -> bool:
        """Whether the model still owes transcript for the audio this burst forwarded."""
        if self.transcript:
            if self.transcript_end_ms is None or self._voiced_end_ms is None:
                return True  # nothing to compare: only the turn ending can settle this burst
            # against the audio that carried sound, since silence is never transcribed, and on the
            # model's own timeline, so neither side moves with network latency
            return self.transcript_end_ms < self._voiced_end_ms
        # a turn the model claimed still owes its text; audio it never claimed is a backchannel
        return self.model_message_id is not None

    def track_audio(self, start_ms: int | None, duration: float, *, voiced: bool) -> None:
        """Advance the playback timeline with a frame that is about to be forwarded."""
        if start_ms is not None:
            if not self.has_audio:
                self._start_ms = start_ms
            elif self._end_ms is not None and start_ms > self._end_ms:
                self._skipped_ms += start_ms - self._end_ms
            self._end_ms = start_ms + round(duration * 1000)
            if voiced:
                self._voiced_end_ms = self._end_ms
        self.has_audio = True

    def playback_time(self, model_ms: int) -> float:
        """Position of a model timestamp in the forwarded audio, in seconds."""
        if self._start_ms is None:
            return 0.0
        return max(0.0, (model_ms - self._start_ms - self._skipped_ms) / 1000)

    def timed_text(self, text: str, start_ms: int | None, end_ms: int | None) -> str:
        """Annotate a transcript fragment with the playback range it is spoken over."""
        if start_ms is None:
            return text
        if self._start_ms is None:
            self._start_ms = start_ms
        # the synchronizer indexes annotations by time, so they must never go backwards
        start_time = max(self._last_annotation, self.playback_time(start_ms))
        end_time = max(start_time, self.playback_time(end_ms)) if end_ms is not None else None
        self._last_annotation = end_time if end_time is not None else start_time
        return TimedString(
            text, start_time=start_time, end_time=end_time if end_time is not None else NOT_GIVEN
        )

    def close(self) -> None:
        if not self.text_ch.closed:
            self.text_ch.close()
        if not self.audio_ch.closed:
            self.audio_ch.close()
        if not self.modalities.done():
            self.modalities.set_result(["audio", "text"])
        self.function_ch.close()
        self.message_ch.close()


class DuplexRealtimeAdapter(RealtimeModel):
    """Runs a :class:`DuplexModel` inside an ``AgentSession``.

    Segments the model's continuous output into turns and presents them as an ordinary
    ``RealtimeSession``, so the voice pipeline needs no duplex-specific path. Output the model never
    transcribes still plays, it simply produces no chat item.
    """

    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        gate: Callable[[], AudioGate] = AdaptiveNoiseGate,
        stalled_transcript_timeout: float = _STALLED_TRANSCRIPT_TIMEOUT,
    ) -> None:
        caps: DuplexCapabilities = duplex_model.capabilities
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=caps.user_transcription,
                auto_tool_reply_generation=caps.auto_tool_reply_generation,
                audio_output=True,
                manual_function_calls=False,
                server_barge_in=True,
                mutable_chat_context=caps.mutable_chat_context,
                mutable_instructions=caps.mutable_instructions,
                mutable_tools=caps.mutable_tools,
                per_response_tool_choice=False,
                supports_say=False,
                manual_response_creation=caps.manual_response_creation,
            )
        )
        self._duplex_model = duplex_model
        self._gate = gate
        self._stalled_timeout = stalled_transcript_timeout

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def model(self) -> str:
        return self._duplex_model.model

    @property
    def provider(self) -> str:
        return self._duplex_model.provider

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # turn detection is inherent to a duplex model, so it is never asked to be off
        return _DuplexRealtimeSession(
            self, self._duplex_model.session(), self._gate(), self._stalled_timeout
        )

    async def aclose(self) -> None:
        await self._duplex_model.aclose()


class _DuplexRealtimeSession(RealtimeSession):
    def __init__(
        self,
        adapter: DuplexRealtimeAdapter,
        duplex: DuplexSession,
        gate: AudioGate,
        stalled_timeout: float,
    ) -> None:
        super().__init__(adapter)
        self._duplex = duplex
        self._gate = gate
        self._stalled_timeout = stalled_timeout
        self._burst: _Burst | None = None
        self._close_handle: asyncio.TimerHandle | None = None
        self._sound_stopped = False
        self._closed_audio_end_ms: int | None = None
        self._open_turns: set[str] = set()

        duplex.on("transcript_delta", self._on_transcript_delta)
        duplex.on("turn_started", self._on_turn_started)
        duplex.on("turn_ended", self._on_turn_ended)
        duplex.on("function_call", self._on_function_call)
        duplex.on("session_reconnected", self._on_session_reconnected)
        for event in (
            "input_speech_started",
            "input_speech_stopped",
            "input_audio_transcription_completed",
            "metrics_collected",
            "error",
        ):
            duplex.on(event, self._forward(event))

        self._segment_atask = asyncio.create_task(
            self._segment_task(), name="DuplexRealtimeSession.segment"
        )

    def _forward(self, event: str) -> Callable[[object], None]:
        def _emit(ev: object) -> None:
            self.emit(event, ev)

        return _emit

    # -- segmenter -------------------------------------------------------------------------

    async def _segment_task(self) -> None:
        try:
            async for f in self._duplex.audio_stream:
                self._on_audio_frame(f)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("duplex audio stream failed")
        finally:
            self._close_burst()

    def _on_audio_frame(self, f: DuplexAudioFrame) -> None:
        # every frame keeps the floor tracking; a tagged one also holds the gate through its tail
        voiced = self._gate.update(f.frame, forced=f.message_id is not None)

        if f.message_id is not None:
            burst = self._burst
            # an orphaned burst holds a dead turn's text, which would prefix this one
            if burst is not None and (
                burst.orphaned or burst.model_message_id not in (None, f.message_id)
            ):
                self._close_burst()
                burst = None
            if burst is None:
                burst = self._open_burst()
            burst.model_message_id = f.message_id
            self._feed(burst, f)
            return

        if self._burst is not None:
            # untagged audio under an open turn or gate is this burst's unlabelled tail
            if voiced or self._open_turns:
                self._feed(self._burst, f)
            else:
                self._sound_stopped = True
                self._maybe_close()
            return

        if voiced:
            self._feed(self._open_burst(), f)

    def _feed(self, burst: _Burst, f: DuplexAudioFrame) -> None:
        self._sound_stopped = False
        self._cancel_close()
        burst.track_audio(f.start_ms, f.frame.duration, voiced=self._gate.voiced)
        if not burst.modalities.done():
            burst.modalities.set_result(["audio", "text"])
        if not burst.audio_ch.closed:
            burst.audio_ch.send_nowait(f.frame)

    def _open_burst(self, *, orphaned: bool = False) -> _Burst:
        burst = _Burst(
            id=shortuuid("item_"),
            orphaned=orphaned,
            message_ch=aio.Chan(),
            function_ch=aio.Chan(),
            text_ch=aio.Chan(),
            audio_ch=aio.Chan(),
            modalities=asyncio.Future(),
        )
        self._burst = burst
        burst.message_ch.send_nowait(
            MessageGeneration(
                message_id=burst.id,
                text_stream=burst.text_ch,
                audio_stream=burst.audio_ch,
                modalities=burst.modalities,
            )
        )
        self.emit(
            "generation_created",
            GenerationCreatedEvent(
                message_stream=burst.message_ch,
                function_stream=burst.function_ch,
                user_initiated=False,
                response_id=burst.id,
            ),
        )
        return burst

    def _maybe_close(self) -> None:
        """Close a burst whose sound has stopped, once the model has finished transcribing it."""
        burst = self._burst
        if burst is None or not self._sound_stopped:
            return

        if burst.turn_ended or not burst.transcript_pending:
            self._close_burst()
        elif self._close_handle is None:
            self._close_handle = asyncio.get_running_loop().call_later(
                self._stalled_timeout, self._close_burst
            )

    def _cancel_close(self) -> None:
        if self._close_handle is not None:
            self._close_handle.cancel()
            self._close_handle = None

    def _close_burst(self) -> None:
        self._cancel_close()
        burst, self._burst = self._burst, None
        if burst is not None:
            self._closed_audio_end_ms = burst._end_ms
            burst.close()

    # -- duplex events ---------------------------------------------------------------------

    def _on_transcript_delta(self, ev: DuplexTranscriptDelta) -> None:
        # only audio defines boundaries and claims the burst's id; fragment ids can differ across
        # one stretch of speech, and claiming one would make the turn's own frames look foreign
        burst = self._burst
        if burst is None:
            # a fragment reaching past the audio already forwarded leads a turn about to start;
            # one that does not has outlived its own audio
            orphaned = (
                ev.end_ms is not None
                and self._closed_audio_end_ms is not None
                and ev.end_ms <= self._closed_audio_end_ms
            )
            if orphaned:
                # emitted rather than dropped, since losing transcript is worse than an odd item
                logger.error(
                    "duplex transcript outlived the audio it describes",
                    extra={
                        "text": ev.text,
                        "message_id": ev.message_id,
                        "span_ms": (ev.start_ms, ev.end_ms),
                        "closed_audio_end_ms": self._closed_audio_end_ms,
                    },
                )
            burst = self._open_burst(orphaned=orphaned)
        burst.transcript += ev.text
        if ev.end_ms is not None:
            burst.transcript_end_ms = max(burst.transcript_end_ms or ev.end_ms, ev.end_ms)
        if not burst.text_ch.closed:
            burst.text_ch.send_nowait(burst.timed_text(ev.text, ev.start_ms, ev.end_ms))
        self._maybe_close()

    def _on_turn_started(self, ev: DuplexTurnStartedEvent) -> None:
        self._open_turns.add(ev.message_id)
        if (burst := self._burst) is None:
            return

        if burst.model_message_id is None:
            # a turn is announced after its first audio, so an unlabelled burst is this one
            burst.model_message_id = ev.message_id
            self._cancel_close()
        elif burst.model_message_id != ev.message_id:
            # whatever is open belongs to the previous turn, the only boundary an
            # untagging model gives
            self._close_burst()

    def _on_turn_ended(self, ev: DuplexTurnEndedEvent) -> None:
        self._open_turns.discard(ev.message_id)
        burst = self._burst
        if burst is not None and burst.model_message_id == ev.message_id:
            # the model is the authority on its turn being over; the gate still ends the sound
            burst.turn_ended = True
            self._maybe_close()

    def _on_function_call(self, call: FunctionCall) -> None:
        burst = self._burst or self._open_burst()
        burst.function_ch.send_nowait(call)

    def _on_session_reconnected(self, ev: object) -> None:
        # a dropped connection never delivers the turn's end, which would hold the burst forever
        self._open_turns.clear()
        self._close_burst()
        self.emit("session_reconnected", ev)

    # -- RealtimeSession -------------------------------------------------------------------

    @property
    def duplex_session(self) -> DuplexSession:
        """The wrapped session, for provider-specific events the adapter does not forward."""
        return self._duplex

    @property
    def chat_ctx(self) -> ChatContext:
        return self._duplex.chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._duplex.tools

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        # as one unit: an immutable configuration must be complete before the first outbound event
        try:
            await self._duplex._update_session(
                instructions=instructions, chat_ctx=chat_ctx, tools=tools
            )
        except RealtimeError:
            logger.exception("failed to configure the duplex session")

    async def update_instructions(self, instructions: str) -> None:
        await self._duplex._update_instructions(instructions)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        await self._duplex._update_chat_ctx(chat_ctx)

    async def update_tools(self, tools: list[Tool]) -> None:
        await self._duplex._update_tools(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        self._duplex._update_options(tool_choice=tool_choice)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._duplex.push_audio(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._duplex.push_video(frame)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()
        try:
            self._duplex._generate_reply(
                instructions=instructions, tool_choice=tool_choice, tools=tools
            )
        except RealtimeError as e:
            fut.set_exception(e)
        return fut

    def commit_audio(self) -> None:
        pass  # input is consumed continuously, there is no buffer to commit

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass  # barge-in is the model's own, and it cannot be cancelled

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass  # the model owns its output timeline

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._segment_atask)
        self._close_burst()
        with contextlib.suppress(Exception):
            await self._duplex.aclose()
