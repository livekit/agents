from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr, TimedString
from ..utils import aio, is_given, shortuuid
from .chat_context import ChatContext, ChatMessage, FunctionCall
from .duplex import (
    DuplexAudioFrame,
    DuplexCapabilities,
    DuplexModel,
    DuplexOutputTranscriptDelta,
    DuplexSession,
)
from .realtime import (
    GenerationCreatedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeError,
    RealtimeModel,
    RealtimeSession,
)
from .tool_context import Tool, ToolChoice, ToolContext
from .utils import compute_chat_ctx_diff

# a floor this low is digital silence; it keeps the gate's ratios finite when a model emits zeros
_SILENCE_FLOOR = 1e-4

# transcript is text the model says it spoke; when no sound ever opens a burst for it, this much
# audio later it is emitted as a text-only generation rather than lost. audio is never held for it
_UNCLAIMED_TRANSCRIPT_MS = 3000

# how long a requested reply waits for the model to start speaking before it counts as declined
_REPLY_TIMEOUT = 10.0

# how far ahead of its sound a fragment is handed over: span labels sit about a frame early
# against the audible onset, and text a little early reads better than late
_ATTACH_LEAD_MS = 300


class AudioGate(Protocol):
    """Decides which frames of a continuously-emitting model carry output worth playing."""

    def update(self, frame: rtc.AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
        ...


class AdaptiveNoiseGate:
    """Opens on output that stands out from the model's own noise floor.

    Thresholds are ratios against the quietest recent frame and durations count audio rather than
    wall clock, so one set of defaults ports across providers, frame sizes and networks.
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

    def update(self, frame: rtc.AudioFrame) -> bool:
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples)))) / 32768.0 if samples.size else 0.0

        self._history.append((rms, frame.duration))
        self._history_duration += frame.duration
        while self._history_duration > self._window and len(self._history) > 1:
            self._history_duration -= self._history.popleft()[1]
        floor = max(min(level for level, _ in self._history), _SILENCE_FLOOR)

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
    """One stretch of audible model output, presented to the framework as a generation.

    It forwards the audio as it arrives and attaches each transcript fragment once its sound is
    reached, placed on the forwarded audio for the synchronizer.
    """

    id: str
    message_ch: aio.Chan[MessageGeneration]
    function_ch: aio.Chan[FunctionCall]
    text_ch: aio.Chan[str]
    audio_ch: aio.Chan[rtc.AudioFrame]
    audio_start_ms: int
    """Where on the adapter's audio clock the burst opened."""
    anchor_ms: int | None = None
    """Span clock minus audio clock, fixed by the first fragment: the sound that opened the gate
    and the oldest unclaimed fragment describe the same moment."""
    opened_at: float = field(default_factory=time.time)
    transcript: str = ""
    _last_annotation: float = 0.0

    def attach(self, fragment: DuplexOutputTranscriptDelta) -> None:
        self.transcript += fragment.text
        text: str = fragment.text
        if fragment.start_ms is not None and self.anchor_ms is not None:
            # placed on the forwarded audio so the synchronizer paces against real speech; the
            # annotations never go backwards, since it indexes them by time
            offset = self.anchor_ms + self.audio_start_ms
            start = max(self._last_annotation, (fragment.start_ms - offset) / 1000)
            end = (
                max(start, (fragment.end_ms - offset) / 1000)
                if fragment.end_ms is not None
                else None
            )
            self._last_annotation = start if end is None else end
            text = TimedString(
                fragment.text, start_time=start, end_time=NOT_GIVEN if end is None else end
            )
        if not self.text_ch.closed:
            self.text_ch.send_nowait(text)

    def close(self) -> None:
        if not self.text_ch.closed:
            self.text_ch.close()
        if not self.audio_ch.closed:
            self.audio_ch.close()
        self.function_ch.close()
        self.message_ch.close()


class DuplexRealtimeAdapter(RealtimeModel):
    """Runs a :class:`DuplexModel` inside an ``AgentSession``.

    Segments the model's continuous output into generations and presents them as an ordinary
    ``RealtimeSession``; output the model never transcribes still plays, it just has no chat item.
    """

    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        gate: Callable[[], AudioGate] = AdaptiveNoiseGate,
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
                paced_audio_output=True,
                mutable_chat_context=caps.mutable_chat_context,
                mutable_instructions=caps.mutable_instructions,
                mutable_tools=caps.mutable_tools,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )
        self._duplex_model = duplex_model
        self._gate = gate

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
        return _DuplexRealtimeSession(self, self._duplex_model.session(), self._gate())

    async def aclose(self) -> None:
        await self._duplex_model.aclose()


class _DuplexRealtimeSession(RealtimeSession):
    def __init__(
        self, adapter: DuplexRealtimeAdapter, duplex: DuplexSession, gate: AudioGate
    ) -> None:
        super().__init__(adapter)
        self._duplex = duplex
        self._gate = gate
        self._burst: _Burst | None = None
        # the adapter's clock: output audio heard so far, which is gapless and real-time
        self._audio_ms = 0
        # the model's words waiting for the sound that carries them, and since when
        self._fragments: deque[DuplexOutputTranscriptDelta] = deque()
        self._waiting_since_ms = 0
        # user_initiated has to be settled before a burst's event goes out, or the framework
        # schedules it as a turn of the model's own as well
        self._pending_reply: asyncio.Future[GenerationCreatedEvent] | None = None
        # the conversation as the framework sees it: the adapter names every message, so this is
        # what a context update diffs against, and only what is new reaches the model
        self._chat_ctx = ChatContext.empty()

        duplex.on("transcript_delta", self._on_transcript_delta)
        duplex.on("function_call", self._on_function_call)
        duplex.on("session_reconnected", self._on_session_reconnected)
        duplex.on("input_audio_transcription_completed", self._on_input_transcription)
        duplex.on("input_speech_started", lambda ev: self.emit("input_speech_started", ev))
        duplex.on("input_speech_stopped", lambda ev: self.emit("input_speech_stopped", ev))
        duplex.on("metrics_collected", lambda ev: self.emit("metrics_collected", ev))
        duplex.on("error", lambda ev: self.emit("error", ev))

        self._segment_atask = asyncio.create_task(
            self._segment_task(), name="DuplexRealtimeSession.segment"
        )

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
        if f.start_ms is not None:
            self._audio_ms = f.start_ms
        # the gate is the one boundary: a burst is open exactly while the model is audibly
        # producing output, its own pauses inside an utterance included
        if self._gate.update(f.frame):
            burst = self._burst or self._open_burst()
            if not burst.audio_ch.closed:
                burst.audio_ch.send_nowait(f.frame)
            self._audio_ms += round(f.frame.duration * 1000)

            # the first fragment anchors the span clock to the audio clock; a later one is due when
            # the audio reaches its span, and one this burst never reaches waits for the next
            while self._fragments:
                fragment = self._fragments[0]
                if fragment.start_ms is not None:
                    if burst.anchor_ms is None:
                        burst.anchor_ms = fragment.start_ms - burst.audio_start_ms
                    if fragment.start_ms - burst.anchor_ms > self._audio_ms + _ATTACH_LEAD_MS:
                        break
                burst.attach(self._fragments.popleft())
            return

        self._audio_ms += round(f.frame.duration * 1000)
        if self._burst is not None:
            self._close_burst()
        elif (
            self._fragments and self._audio_ms - self._waiting_since_ms >= _UNCLAIMED_TRANSCRIPT_MS
        ):
            logger.error(
                "duplex transcript outlived the audio it describes",
                extra={"text": "".join(f.text for f in self._fragments)},
            )
            burst = self._open_burst()
            while self._fragments:
                burst.attach(self._fragments.popleft())
            self._close_burst()

    def _open_burst(self, *, message: bool = True) -> _Burst:
        burst = self._burst = _Burst(
            id=shortuuid("item_"),
            message_ch=aio.Chan(),
            function_ch=aio.Chan(),
            text_ch=aio.Chan(),
            audio_ch=aio.Chan(),
            audio_start_ms=self._audio_ms,
        )
        ev = GenerationCreatedEvent(
            message_stream=burst.message_ch,
            function_stream=burst.function_ch,
            user_initiated=False,
            response_id=burst.id,
        )
        # the model answers on the one stream it has, so speech opening is the reply asked for; a
        # call on its own is not, the speech it leads to is
        if message and self._pending_reply is not None and not self._pending_reply.done():
            ev.user_initiated = True
            self._pending_reply.set_result(ev)
        self.emit("generation_created", ev)
        if message:
            modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
            modalities.set_result(["audio", "text"])
            burst.message_ch.send_nowait(
                MessageGeneration(
                    message_id=burst.id,
                    text_stream=burst.text_ch,
                    audio_stream=burst.audio_ch,
                    modalities=modalities,
                )
            )
        return burst

    def _close_burst(self) -> None:
        burst, self._burst = self._burst, None
        if burst is not None:
            burst.close()
            if burst.transcript:
                # under the id and time the framework will use for it, so a context update matches
                self._chat_ctx.insert(
                    ChatMessage(
                        id=burst.id,
                        role="assistant",
                        content=[burst.transcript],
                        created_at=burst.opened_at,
                    )
                )
        self._waiting_since_ms = self._audio_ms

    def _on_transcript_delta(self, ev: DuplexOutputTranscriptDelta) -> None:
        # attached on the next frame: the sound places the words, and the frames never stop
        if not self._fragments:
            self._waiting_since_ms = self._audio_ms
        self._fragments.append(ev)

    def _on_input_transcription(self, ev: InputTranscriptionCompleted) -> None:
        if ev.is_final:
            self._chat_ctx.insert(
                ChatMessage(
                    id=ev.item_id,
                    role="user",
                    content=[ev.transcript],
                    transcript_confidence=ev.confidence if ev.confidence is not None else 1.0,
                    created_at=ev.turn_started_at
                    if ev.turn_started_at is not None
                    else time.time(),
                )
            )
        self.emit("input_audio_transcription_completed", ev)

    def _on_function_call(self, call: FunctionCall) -> None:
        self._chat_ctx.insert(call)
        # a call joins the burst in flight so the tool runs while the model talks; alone, it is a
        # generation of its own, over as soon as it is delivered
        if self._burst is not None:
            self._burst.function_ch.send_nowait(call)
            return
        self._open_burst(message=False).function_ch.send_nowait(call)
        self._close_burst()

    def _on_session_reconnected(self, ev: object) -> None:
        # a dropped connection never delivers the rest of a burst, the sound its waiting words
        # describe, or the reply it was asked for
        self._fragments.clear()
        self._close_burst()
        self._fail_pending_reply("the session reconnected before the model replied")
        self.emit("session_reconnected", ev)

    # RealtimeSession

    @property
    def duplex_session(self) -> DuplexSession:
        """The wrapped session, for provider-specific events the adapter does not forward."""
        return self._duplex

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx.copy()

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
        if is_given(chat_ctx):
            chat_ctx = chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)
            self._chat_ctx = chat_ctx.copy()
        try:
            await self._duplex._update_session(
                instructions=instructions, chat_ctx=chat_ctx, tools=tools
            )
        except RealtimeError:
            logger.exception("failed to configure the duplex session")

    async def update_instructions(self, instructions: str) -> None:
        await self._duplex._update_instructions(instructions)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        from ..voice.generation import remove_instructions

        chat_ctx = chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)
        remove_instructions(chat_ctx)
        diff = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        # the framework records the model's speech as it was played, which may be less than what
        # the model said; anything else it edits or removes, the model has already been told
        if edited := [
            item_id
            for item_id in diff.to_remove + [item_id for _, item_id in diff.to_update]
            if not isinstance(item := self._chat_ctx.get_by_id(item_id), ChatMessage)
            or item.role != "assistant"
        ]:
            logger.warning(
                "duplex context is append-only; the model keeps what it has been told",
                extra={"item_ids": edited},
            )
        if new_items := [
            item for _, item_id in diff.to_create if (item := chat_ctx.get_by_id(item_id))
        ]:
            await self._duplex._append_items(new_items)
        self._chat_ctx = chat_ctx

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

        # the reply is the next speech to open; asking is a request the model may never answer
        self._fail_pending_reply("a newer ask superseded this one")
        self._pending_reply = fut

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(RealtimeError("the model did not start speaking when asked"))

        timeout = asyncio.get_running_loop().call_later(_REPLY_TIMEOUT, _on_timeout)
        fut.add_done_callback(lambda _: timeout.cancel())
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

    def _fail_pending_reply(self, reason: str) -> None:
        # the framework's reply task handles RealtimeError; a cancelled future would end it
        if self._pending_reply is not None and not self._pending_reply.done():
            self._pending_reply.set_exception(RealtimeError(reason))

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._segment_atask)
        self._close_burst()
        self._fail_pending_reply("the session closed before the model replied")
        with contextlib.suppress(Exception):
            await self._duplex.aclose()
