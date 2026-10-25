from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent as CoreGenerationCreatedEvent,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession as CoreRealtimeSession,
    RealtimeSessionReconnectedEvent,
)
from livekit.agents.llm.realtime_fallback_adapter import (
    RealtimeAvailabilityChangedEvent as RealtimeAvailabilityChangedEvent,
    RealtimeModelFallbackAdapter as CoreRealtimeModelFallbackAdapter,
    _FallbackRealtimeSession as CoreFallbackRealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice
from livekit.agents.log import logger
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import aio, is_given

from ._realtime_types import (
    GenerationCreatedEvent,
    can_reuse,
    playback_finished,
    set_input_text,
)

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession


# pipeline-shaping caps that must match across models (set at activity start, can't change mid-call)
_HARD_CAPABILITIES = (
    "audio_output",
    "turn_detection",
    "supports_overlapping_speech",
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
    "can_disable_turn_detection",
)


def _merge_capabilities(models: list[RealtimeModel]) -> RealtimeCapabilities:
    first = models[0].capabilities
    for model in models[1:]:
        caps = model.capabilities
        for name in _HARD_CAPABILITIES:
            if getattr(caps, name, False) != getattr(first, name, False):
                raise ValueError(
                    f"all realtime models must agree on `{name}` to be used in a "
                    f"RealtimeModelFallbackAdapter, got "
                    f"{getattr(first, name, False)} and {getattr(caps, name, False)}"
                )

    merged = {name: getattr(first, name, False) for name in _HARD_CAPABILITIES}
    for name in _SOFT_CAPABILITIES:
        merged[name] = all(getattr(model.capabilities, name, False) for model in models)

    return RealtimeCapabilities(
        **{
            key: value
            for key, value in merged.items()
            if key in RealtimeCapabilities.__dataclass_fields__
        }
    )


class RealtimeModelFallbackAdapter(CoreRealtimeModelFallbackAdapter):
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
                the ``audio_output``, ``turn_detection`` and ``supports_overlapping_speech``
                capabilities.
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
        self._agent_session: AgentSession | None = None
        self._sessions: weakref.WeakSet[CoreFallbackRealtimeSession] = weakref.WeakSet()

        # the model currently serving sessions; used to label metrics & traces
        self._active_instance: RealtimeModel = models[0]

    def session(self, *, turn_detection_disabled: bool = False) -> _FallbackRealtimeSession:
        sess = _FallbackRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        sess._agent_session = getattr(self, "_agent_session", None)
        return sess


class _FallbackRealtimeSession(CoreFallbackRealtimeSession):
    """Bound once by AgentActivity; swaps the inner child session internally."""

    def __init__(
        self, adapter: RealtimeModelFallbackAdapter, *, turn_detection_disabled: bool = False
    ) -> None:
        # The installed core owns event subscriptions, availability, and delegates.
        # These fields extend its lifecycle for Prism's acknowledged operations.
        self._settings_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._initializing = False
        self._initialization_error: Exception | None = None
        super().__init__(adapter, turn_detection_disabled=turn_detection_disabled)

    def _bind(self, child: CoreRealtimeSession[Any]) -> None:
        # Capture the owning child before an active-session swap. The core's
        # subscription and unsubscription paths use these same callback objects.
        def generation_created(event: object) -> None:
            if isinstance(event, CoreGenerationCreatedEvent):
                event = self._track_generation(event, child)
            self.emit("generation_created", event)

        self._forwarders["generation_created"] = generation_created
        if "input_speech_hint" not in self._extra_forwarders:
            self._extra_forwarders["input_speech_hint"] = lambda event: self.emit(
                "input_speech_hint", event
            )
        super()._bind(child)

    def _next_available_index(
        self, *, exclude_current: bool = False, exclude: set[int] | None = None
    ) -> int | None:
        # re-enable models whose cooldown expired, then pick the first available (primary preferred)
        now = time.time()
        for i, deadline in enumerate(self._cooldown_deadline):
            if not self._available[i] and deadline <= now:
                self._set_available(i, True)

        for i in range(len(self._adapter._models)):
            if exclude_current and i == self._active_index:
                continue
            if exclude is not None and i in exclude:
                continue
            if self._available[i]:
                return i
        return None

    def _on_child_error(self, error: RealtimeModelError) -> None:
        if self._initializing and not error.recoverable:
            # Startup owns failure and replacement until configuration settles.
            self._initialization_error = error.error
            return
        super()._on_child_error(error)

    async def _swap(self, target_index: int, was_speaking: bool) -> None:
        """Replace the active child with a fresh session on ``target_index``.

        If the agent was speaking, its reply is interrupted first (committing the heard content to
        the agent chat context) and re-issued on the new session afterwards.
        """
        # interrupt through the AgentSession so playout/state stay coordinated and the heard
        # content is committed (best-effort: the provider may already be dead)
        if self._agent_session is not None:
            try:
                await self._agent_session.interrupt(force=True)
            except Exception:
                logger.debug("failed to interrupt the agent before swap", exc_info=True)

        async with self._swap_lock:
            # replay the agent chat context (what the user heard); fall back to the child's when unbound
            if self._agent_session is not None:
                chat_ctx = self._agent_session.current_agent.chat_ctx
            else:
                chat_ctx = self._active.chat_ctx

            self._swapping = True
            try:
                # close the old child; best-effort since the provider may already be dead
                self._unbind(self._active)
                with contextlib.suppress(Exception):
                    await self._active.aclose()

                error = await self._start_with_fallback(
                    index=target_index,
                    reuse_active=False,
                    instructions=self._instructions,
                    chat_ctx=chat_ctx,
                    tools=self._tools,
                    report_attempt_errors=False,
                )
            finally:
                self._swapping = False

            if error is not None:
                return

            # a swap is a reconnect from the caller's perspective
            self.emit("session_reconnected", RealtimeSessionReconnectedEvent())

            # re-issue the interrupted reply on the new session
            if (
                was_speaking
                and self._adapter._regenerate_on_swap
                and self._agent_session is not None
            ):
                self._agent_session.generate_reply()

    def can_reuse(self, chat_ctx: ChatContext) -> bool:
        return can_reuse(self._active, chat_ctx)

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        if is_given(instructions):
            self._instructions = instructions
        if is_given(tools):
            self._tools = tools
        if self._swap_task is not None and not self._swap_task.done():
            await self._swap_task
        self._swap_task = asyncio.create_task(
            self._initialize(instructions=instructions, chat_ctx=chat_ctx, tools=tools)
        )
        await self._swap_task

    async def _initialize(
        self,
        *,
        instructions: NotGivenOr[str],
        chat_ctx: NotGivenOr[ChatContext],
        tools: NotGivenOr[list[Tool]],
    ) -> None:
        async with self._swap_lock:
            self._swapping = True
            try:
                error = await self._start_with_fallback(
                    index=self._active_index,
                    reuse_active=True,
                    instructions=instructions,
                    chat_ctx=chat_ctx,
                    tools=tools,
                    report_attempt_errors=True,
                )
                if error is not None:
                    raise error
            finally:
                self._swapping = False

    async def _start_with_fallback(
        self,
        *,
        index: int,
        reuse_active: bool,
        instructions: NotGivenOr[str],
        chat_ctx: NotGivenOr[ChatContext],
        tools: NotGivenOr[list[Tool]],
        report_attempt_errors: bool,
    ) -> Exception | None:
        """Start each candidate at most once while the caller holds _swap_lock.

        Initial setup can configure the existing child with omitted settings.
        Replacements receive saved settings and history. Startup reports each
        failure; swaps retain their existing single error on exhaustion.
        """
        history = chat_ctx if is_given(chat_ctx) else self._active.chat_ctx
        if reuse_active:
            # Snapshot before initial configuration can modify the current child.
            history = history.copy()
        attempted: set[int] = set()
        self._initializing = True
        try:
            while True:
                self._initialization_error = None
                replacing = not reuse_active or bool(attempted)
                attempted.add(index)
                try:
                    if replacing:
                        self._active_index = index
                        self._active = self._adapter._models[index].session(
                            turn_detection_disabled=self._turn_detection_disabled
                        )
                        self._bind(self._active)
                        instructions, chat_ctx, tools = self._instructions, history, self._tools
                    await self._active._update_session(
                        instructions=instructions, chat_ctx=chat_ctx, tools=tools
                    )
                    if replacing and is_given(self._tool_choice):
                        self._active.update_options(tool_choice=self._tool_choice)
                    if self._initialization_error is not None:
                        raise self._initialization_error
                    self._adapter._active_instance = self._adapter._models[index]
                    return None
                except Exception as error:
                    self._unbind(self._active)
                    with contextlib.suppress(Exception):
                        await self._active.aclose()
                    self._set_available(index, False)
                    # Cooldown may expire during startup, or be zero. Do not
                    # retry a failed candidate during this operation.
                    target = self._next_available_index(exclude=attempted)
                    if report_attempt_errors or target is None:
                        self.emit(
                            "error",
                            RealtimeModelError(
                                timestamp=time.time(),
                                label=self._adapter.label,
                                error=error,
                                recoverable=target is not None,
                            ),
                        )
                    else:
                        logger.exception("failed to start realtime model on swap, trying next")
                    if target is None:
                        return error
                    index = target
        except asyncio.CancelledError:
            self._unbind(self._active)
            with contextlib.suppress(Exception):
                await self._active.aclose()
            raise
        finally:
            self._initializing = False
            self._initialization_error = None

    async def set_input_text(
        self, text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
    ) -> ChatMessage:
        # Unlike microphone frames, a typed turn must not be discarded during a swap.
        async with self._swap_lock:
            return await set_input_text(self._active, text, message_id=message_id)

    def _track_generation(
        self, event: CoreGenerationCreatedEvent, child: CoreRealtimeSession[Any]
    ) -> GenerationCreatedEvent:
        # The generation carries its owner across swaps, including nested fallback
        # adapters. Provider response and message IDs stay unchanged.
        owner = getattr(event, "playback_session", None) or child
        if isinstance(event, GenerationCreatedEvent):
            return replace(event, playback_session=owner)
        return GenerationCreatedEvent(
            message_stream=event.message_stream,
            function_stream=event.function_stream,
            user_initiated=event.user_initiated,
            response_id=event.response_id,
            playback_session=owner,
        )

    def _track_reply(
        self, future: asyncio.Future[CoreGenerationCreatedEvent], child: CoreRealtimeSession[Any]
    ) -> asyncio.Future[CoreGenerationCreatedEvent]:
        tracked: asyncio.Future[CoreGenerationCreatedEvent] = future.get_loop().create_future()

        def _complete(source: asyncio.Future[CoreGenerationCreatedEvent]) -> None:
            if source.cancelled():
                tracked.cancel()
            elif (error := source.exception()) is not None:
                if not tracked.done():
                    tracked.set_exception(error)
            elif not tracked.done():
                tracked.set_result(self._track_generation(source.result(), child))
            elif tracked.cancelled():
                # Acceptance can race cancellation before this callback runs. The
                # caller never received the generation, so it cannot report skipped
                # playback. Target its owning child and response explicitly.
                try:
                    accepted = source.result()
                    owner = getattr(accepted, "playback_session", None) or child
                    owner.interrupt()
                    playback_finished(
                        owner,
                        response_id=accepted.response_id,
                        message_id=None,
                        audio_end_ms=0,
                        interrupted=True,
                        interrupted_by_user=False,
                    )
                except Exception:
                    logger.exception("failed to report cancelled realtime reply playback")

        def _cancel(source: asyncio.Future[CoreGenerationCreatedEvent]) -> None:
            if source.cancelled():
                future.cancel()

        future.add_done_callback(_complete)
        tracked.add_done_callback(_cancel)
        return tracked

    def playback_finished(
        self,
        *,
        response_id: str | None,
        message_id: str | None,
        audio_end_ms: int,
        interrupted: bool,
        interrupted_by_user: bool,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
        modalities: list[Literal["text", "audio"]] | None = None,
    ) -> None:
        # Direct callers address the active session. AgentActivity instead uses
        # GenerationCreatedEvent.playback_session to retain ownership across swaps.
        playback_finished(
            self._active,
            response_id=response_id,
            message_id=message_id,
            audio_end_ms=audio_end_ms,
            interrupted=interrupted,
            interrupted_by_user=interrupted_by_user,
            audio_transcript=audio_transcript,
            modalities=modalities,
        )

    async def update_instructions(self, instructions: str) -> None:
        await self._update_settings(instructions=instructions)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        if self._closed:
            raise RuntimeError("Realtime fallback session is closed")
        target = chat_ctx.copy()

        async def apply() -> None:
            async with self._swap_lock:
                if self._closed:
                    raise RuntimeError("Realtime fallback session is closed")
                await self._active.update_chat_ctx(target)

        task = asyncio.create_task(apply())
        self._settings_tasks.add(task)
        task.add_done_callback(self._settings_tasks.discard)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def update_tools(self, tools: list[Tool]) -> None:
        await self._update_settings(tools=list(tools))

    async def _update_settings(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        if self._closed:
            raise RuntimeError("Realtime fallback session is closed")

        async def apply() -> None:
            async with self._swap_lock:
                if is_given(instructions):
                    await self._active.update_instructions(instructions)
                    self._instructions = instructions
                if is_given(tools):
                    await self._active.update_tools(tools)
                    self._tools = tools

        # Keep the acknowledged child settings and the restart snapshot together,
        # including when a caller cancels while the child waits for its acknowledgment.
        task = asyncio.create_task(apply())
        self._settings_tasks.add(task)
        task.add_done_callback(self._settings_tasks.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[CoreGenerationCreatedEvent]:
        child = self._active
        return self._track_reply(
            child.generate_reply(instructions=instructions, tool_choice=tool_choice, tools=tools),
            child,
        )

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[CoreGenerationCreatedEvent]:
        child = self._active
        return self._track_reply(child.say(text), child)

    async def aclose(self) -> None:
        # cancel an in-flight swap first, else its fresh child would leak past aclose
        self._closed = True
        await aio.cancel_and_wait(*self._settings_tasks)
        if self._swap_task is not None:
            await aio.cancel_and_wait(self._swap_task)
        self._unbind(self._active)
        await self._active.aclose()
