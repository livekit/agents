from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext
from .realtime import (
    EventTypes,
    GenerationCreatedEvent,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
    RealtimeSessionReconnectedEvent,
)
from .tool_context import Tool, ToolChoice, ToolContext

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession


@dataclass
class RealtimeAvailabilityChangedEvent:
    realtime_model: RealtimeModel
    available: bool


# pipeline-shaping caps that must match across models (set at activity start, can't change mid-call)
_HARD_CAPABILITIES = (
    "audio_output",
    "turn_detection",
)

# caps exposed as the conservative AND; the active model's exact value is read per-turn from the session
_SOFT_CAPABILITIES = (
    "message_truncation",
    "user_transcription",
    "manual_function_calls",
    "auto_tool_reply_generation",
    "mutable_chat_context",
    "mutable_instructions",
    "mutable_tools",
    "per_response_tool_choice",
    "supports_say",
)

# child events re-emitted on the wrapper
_FORWARDED_EVENTS: tuple[EventTypes, ...] = (
    "input_speech_started",
    "input_speech_stopped",
    "input_audio_transcription_completed",
    "generation_created",
    "session_reconnected",
    "metrics_collected",
    "remote_item_added",
)


def _merge_capabilities(models: list[RealtimeModel]) -> RealtimeCapabilities:
    first = models[0].capabilities
    for model in models[1:]:
        caps = model.capabilities
        for name in _HARD_CAPABILITIES:
            if getattr(caps, name) != getattr(first, name):
                raise ValueError(
                    f"all realtime models must agree on `{name}` to be used in a "
                    f"RealtimeModelFallbackAdapter, got "
                    f"{getattr(first, name)} and {getattr(caps, name)}"
                )

    merged = {name: getattr(first, name) for name in _HARD_CAPABILITIES}
    for name in _SOFT_CAPABILITIES:
        merged[name] = all(getattr(model.capabilities, name) for model in models)

    return RealtimeCapabilities(**merged)


class RealtimeModelFallbackAdapter(
    RealtimeModel,
    rtc.EventEmitter[Literal["realtime_availability_changed"]],
):
    """Falls back between realtime models (or restarts one), preserving chat context and handlers."""

    def __init__(
        self,
        models: list[RealtimeModel],
        *,
        cooldown: float = 10.0,
        regenerate_on_swap: bool = True,
    ) -> None:
        """Fall back between realtime models while preserving chat context.

        Args:
            models: Ordered models; the first is primary, the rest fallbacks. All must agree on
                the ``audio_output`` and ``turn_detection`` capabilities.
            cooldown: Seconds a failed model stays unavailable before it can be preferred again.
            regenerate_on_swap: Re-issue the reply on the new session if one was in progress.

        Raises:
            ValueError: If no models are given or their hard capabilities disagree.
        """
        if len(models) < 1:
            raise ValueError("at least one RealtimeModel instance must be provided.")

        RealtimeModel.__init__(self, capabilities=_merge_capabilities(models))
        rtc.EventEmitter.__init__(self)
        self._models = models
        self._cooldown = cooldown
        self._regenerate_on_swap = regenerate_on_swap
        self._sessions: weakref.WeakSet[_FallbackRealtimeSession] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "RealtimeModelFallbackAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self) -> _FallbackRealtimeSession:
        sess = _FallbackRealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def restart_session(self, *, switch_model: bool = False) -> None:
        """Bring up a fresh underlying session, preserving chat context and bound handlers.

        Args:
            switch_model: Bring the new session up on the next available model instead of the
                current one.
        """
        for sess in list(self._sessions):
            await sess.restart(switch_model=switch_model)

    async def aclose(self) -> None:
        for model in self._models:
            await model.aclose()


class _FallbackRealtimeSession(RealtimeSession[Literal["realtime_availability_changed"]]):
    """Bound once by AgentActivity; swaps the inner child session internally."""

    def __init__(self, adapter: RealtimeModelFallbackAdapter) -> None:
        super().__init__(adapter)
        self._adapter = adapter

        # session state replayed onto a new child on swap
        self._instructions: NotGivenOr[str] = NOT_GIVEN
        self._tools: NotGivenOr[list[Tool]] = NOT_GIVEN
        self._tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN

        # stable per-event forwarders so they can be detached on swap
        def _make_forwarder(event: EventTypes) -> Callable[[object], None]:
            # callbacks receive only the payload, so bind the event name per forwarder
            def _forward(ev: object) -> None:
                self.emit(event, ev)

            return _forward

        self._forwarders: dict[EventTypes, Callable[[object], None]] = {
            event: _make_forwarder(event) for event in _FORWARDED_EVENTS
        }

        # per-model availability, with a cooldown after a failure
        self._available = [True] * len(adapter._models)
        self._cooldown_deadline = [0.0] * len(adapter._models)

        self._swap_task: asyncio.Task[None] | None = None
        self._swap_lock = asyncio.Lock()
        # bound by AgentActivity; used to read agent state and drive interrupt/generate_reply on swap
        self._agent_session: AgentSession | None = None

        # audio during a swap is dropped; replaying it would lag the model behind realtime
        self._swapping = False

        self._active_index = 0
        self._active = adapter._models[0].session()
        self._bind(self._active)

    def _bind(self, child: RealtimeSession) -> None:
        for event, forwarder in self._forwarders.items():
            child.on(event, forwarder)
        child.on("error", self._on_child_error)

    def _unbind(self, child: RealtimeSession) -> None:
        for event, forwarder in self._forwarders.items():
            child.off(event, forwarder)
        child.off("error", self._on_child_error)

    def _set_available(self, index: int, available: bool) -> None:
        if self._available[index] == available:
            return
        self._available[index] = available
        self._adapter.emit(
            "realtime_availability_changed",
            RealtimeAvailabilityChangedEvent(
                realtime_model=self._adapter._models[index], available=available
            ),
        )

    def _next_available_index(self, *, exclude_current: bool = False) -> int | None:
        # re-enable models whose cooldown expired, then pick the first available (primary preferred)
        now = time.time()
        for i, deadline in enumerate(self._cooldown_deadline):
            if not self._available[i] and deadline <= now:
                self._set_available(i, True)

        for i in range(len(self._adapter._models)):
            if exclude_current and i == self._active_index:
                continue
            if self._available[i]:
                return i
        return None

    def _is_agent_speaking(self) -> bool:
        # "thinking" (generating) or "speaking" (playing out) both mean a reply is in progress
        return self._agent_session is not None and self._agent_session.agent_state in (
            "speaking",
            "thinking",
        )

    def _on_child_error(self, error: RealtimeModelError) -> None:
        if error.recoverable:
            # surface it and let the plugin's own reconnect handle it
            self.emit("error", error)
            return

        # mark the dead model unavailable for a cooldown, then find a fallback
        self._set_available(self._active_index, False)
        self._cooldown_deadline[self._active_index] = time.time() + self._adapter._cooldown
        target = self._next_available_index()
        if target is None:
            # exhausted: escalate so AgentSession can close
            self.emit("error", error)
            return

        # recoverable while a fallback remains, so the session isn't torn down
        self.emit("error", error.model_copy(update={"recoverable": True}))
        if self._swap_task is None or self._swap_task.done():
            # capture the speaking state now; the dead generation may flip it before the swap runs
            self._swap_task = asyncio.create_task(self._swap(target, self._is_agent_speaking()))

    async def restart(self, *, switch_model: bool) -> None:
        """Restart the underlying session, optionally on the next available model."""
        if switch_model:
            # fall back to the current model if no other is available
            target = self._next_available_index(exclude_current=True)
            target = self._active_index if target is None else target
        else:
            target = self._active_index
        await self._swap(target, self._is_agent_speaking())

    async def _swap(self, target_index: int, was_speaking: bool) -> None:
        """Replace the active child with a fresh session on ``target_index``.

        If the agent was speaking, its reply is interrupted first (committing the heard content to
        the agent chat context) and re-issued on the new session afterwards.
        """
        async with self._swap_lock:
            # interrupt through the AgentSession so playout/state stay coordinated and the heard
            # content is committed (best-effort: the provider may already be dead)
            if self._agent_session is not None:
                try:
                    await self._agent_session.interrupt(force=True)
                except Exception:
                    logger.debug("failed to interrupt the agent before swap", exc_info=True)

            # replay the agent chat context (what the user heard); fall back to the child's when unbound
            if self._agent_session is not None:
                chat_ctx = self._agent_session.current_agent.chat_ctx
            else:
                chat_ctx = self._active.chat_ctx

            self._swapping = True
            try:
                self._unbind(self._active)
                await self._active.aclose()

                self._active = self._adapter._models[target_index].session()
                self._active_index = target_index
                self._bind(self._active)

                await self._active._update_session(
                    instructions=self._instructions, chat_ctx=chat_ctx, tools=self._tools
                )
                if is_given(self._tool_choice):
                    self._active.update_options(tool_choice=self._tool_choice)
            except Exception as e:
                # a failed swap would otherwise wedge the session silently; escalate so the
                # AgentSession can close instead of continuing on a broken session
                logger.exception("failed to swap the realtime session")
                self.emit(
                    "error",
                    RealtimeModelError(
                        timestamp=time.time(), label=self._adapter.label, error=e, recoverable=False
                    ),
                )
                return
            finally:
                self._swapping = False

            # a swap is a reconnect from the caller's perspective
            self.emit("session_reconnected", RealtimeSessionReconnectedEvent())

            # re-issue the interrupted reply on the new session
            if (
                was_speaking
                and self._adapter._regenerate_on_swap
                and self._agent_session is not None
            ):
                self._agent_session.generate_reply()

    @property
    def capabilities(self) -> RealtimeCapabilities:
        # the active model's caps, so per-turn consumers see the model actually in use
        return self._active.realtime_model.capabilities

    @property
    def chat_ctx(self) -> ChatContext:
        return self._active.chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._active.tools

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        await self._active.update_instructions(instructions)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        if self._swapping:
            # dropped; the swap replays the agent chat context afterwards
            return
        await self._active.update_chat_ctx(chat_ctx)

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tools = tools
        await self._active.update_tools(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        self._tool_choice = tool_choice
        self._active.update_options(tool_choice=tool_choice)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._swapping:
            # drop during swap; replaying would lag the model
            return
        self._active.push_audio(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        self._active.push_video(frame)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        return self._active.generate_reply(
            instructions=instructions, tool_choice=tool_choice, tools=tools
        )

    def commit_audio(self) -> None:
        self._active.commit_audio()

    def clear_audio(self) -> None:
        self._active.clear_audio()

    def interrupt(self) -> None:
        self._active.interrupt()

    def start_user_activity(self) -> None:
        self._active.start_user_activity()

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[GenerationCreatedEvent]:
        return self._active.say(text)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._active.truncate(
            message_id=message_id,
            modalities=modalities,
            audio_end_ms=audio_end_ms,
            audio_transcript=audio_transcript,
        )

    async def aclose(self) -> None:
        self._unbind(self._active)
        await self._active.aclose()
