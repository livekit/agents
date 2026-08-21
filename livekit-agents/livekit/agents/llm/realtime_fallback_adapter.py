from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import aio, is_given
from .chat_context import ChatContext, MetricsMetadata
from .realtime import (
    EventTypes,
    GenerationCreatedEvent,
    RealtimeCapabilities,
    RealtimeError,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
    RealtimeSessionReconnectedEvent,
    _UserMessageSyncResult,
    _UserMessageSyncStatus,
)
from .tool_context import Tool, ToolChoice, ToolContext

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession


@dataclass
class RealtimeAvailabilityChangedEvent:
    realtime_model: RealtimeModel
    available: bool


_SwapPhase = Literal["idle", "interrupting", "replacing"]


@dataclass(frozen=True)
class _SwapOutcome:
    child_epoch: int
    replayed_item_ids: frozenset[str]
    error: RealtimeError | None = None


@dataclass(frozen=True)
class _PendingCrossSwapGeneration:
    completion: asyncio.Future[_SwapOutcome] | None
    chat_ctx: ChatContext
    message_id: str


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
    "can_disable_turn_detection",
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

        # the model currently serving sessions; used to label metrics & traces
        self._active_instance: RealtimeModel = models[0]

    @property
    def model(self) -> str:
        return "RealtimeModelFallbackAdapter"

    @property
    def provider(self) -> str:
        return "livekit"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata of the model currently serving sessions (the primary until a swap)."""
        return self._active_instance.metrics_metadata

    def session(self, *, turn_detection_disabled: bool = False) -> _FallbackRealtimeSession:
        sess = _FallbackRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
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


class _FallbackRealtimeSession(RealtimeSession[str]):
    """Bound once by AgentActivity; swaps the inner child session internally."""

    def __init__(
        self, adapter: RealtimeModelFallbackAdapter, *, turn_detection_disabled: bool = False
    ) -> None:
        super().__init__(adapter)
        self._adapter = adapter
        # applied to every underlying session, including those brought up on a swap
        self._turn_detection_disabled = turn_detection_disabled

        # session state replayed onto a new child on swap
        self._instructions: NotGivenOr[str] = NOT_GIVEN
        self._tools: NotGivenOr[list[Tool]] = NOT_GIVEN
        self._tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
        self._provider_replay_excluded_item_ids: set[str] = set()

        # stable per-event forwarders so they can be detached on swap
        def _make_forwarder(event: EventTypes) -> Callable[[object], None]:
            # callbacks receive only the payload, so bind the event name per forwarder
            def _forward(ev: object) -> None:
                self.emit(event, ev)

            return _forward

        self._forwarders: dict[EventTypes, Callable[[object], None]] = {
            event: _make_forwarder(event) for event in _FORWARDED_EVENTS
        }
        self._extra_forwarders: dict[str, Callable[..., None]] = {}

        # per-model availability, with a cooldown after a failure
        self._available = [True] * len(adapter._models)
        self._cooldown_deadline = [0.0] * len(adapter._models)

        self._swap_task: asyncio.Task[None] | None = None
        self._swap_serialization_lock = asyncio.Lock()
        self._swap_state_lock = asyncio.Lock()
        self._swap_phase: _SwapPhase = "idle"
        self._swap_completion: asyncio.Future[_SwapOutcome] | None = None
        self._child_epoch = 0
        self._pending_cross_swap_generations: dict[
            asyncio.Task[Any], _PendingCrossSwapGeneration
        ] = {}
        # bound by AgentActivity; used to read agent state and drive interrupt/generate_reply on swap
        self._agent_session: AgentSession | None = None

        self._active_index = 0
        self._active_bound = False
        self._active = adapter._models[0].session(
            turn_detection_disabled=self._turn_detection_disabled
        )
        # a fresh session always starts on the primary, even after an earlier failover
        adapter._active_instance = adapter._models[0]
        self._bind(self._active)

    def _bind(self, child: RealtimeSession) -> None:
        for event, forwarder in self._forwarders.items():
            child.on(event, forwarder)
        for extra_event, forwarder in self._extra_forwarders.items():
            child.on(extra_event, forwarder)
        child.on("error", self._on_child_error)
        self._active_bound = True

    def _unbind(self, child: RealtimeSession) -> None:
        self._active_bound = False
        for event, forwarder in self._forwarders.items():
            child.off(event, forwarder)
        for extra_event, forwarder in self._extra_forwarders.items():
            child.off(extra_event, forwarder)
        child.off("error", self._on_child_error)

    def on(
        self,
        event: EventTypes | str,
        callback: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        if event not in _FORWARDED_EVENTS and event != "error":
            forwarder = self._extra_forwarders.get(event)
            if forwarder is None:

                def _forward(*args: object) -> None:
                    self.emit(event, *args)

                forwarder = _forward
                self._extra_forwarders[event] = forwarder
                if self._active_bound:
                    self._active.on(event, forwarder)

        return super().on(event, callback)

    def _set_available(self, index: int, available: bool) -> None:
        if self._available[index] == available:
            return
        self._available[index] = available
        if not available:
            # keep the model out of rotation until the cooldown expires
            self._cooldown_deadline[index] = time.time() + self._adapter._cooldown
        self._adapter.emit(
            "realtime_availability_changed",
            RealtimeAvailabilityChangedEvent(
                realtime_model=self._adapter._models[index], available=available
            ),
        )

    def _next_available_index(
        self,
        *,
        exclude_current: bool = False,
        excluded_indices: set[int] | None = None,
    ) -> int | None:
        # re-enable models whose cooldown expired, then pick the first available (primary preferred)
        now = time.time()
        for i, deadline in enumerate(self._cooldown_deadline):
            if not self._available[i] and deadline <= now:
                self._set_available(i, True)

        for i in range(len(self._adapter._models)):
            if exclude_current and i == self._active_index:
                continue
            if excluded_indices is not None and i in excluded_indices:
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

    def _observe_automatic_swap(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            self.emit(
                "error",
                RealtimeModelError(
                    timestamp=time.time(),
                    label=self._adapter.label,
                    error=error,
                    recoverable=False,
                ),
            )

    def _on_child_error(self, error: RealtimeModelError) -> None:
        if error.recoverable:
            # surface it and let the plugin's own reconnect handle it
            self.emit("error", error)
            return

        # mark the dead model unavailable for a cooldown, then find a fallback
        self._set_available(self._active_index, False)
        target = self._next_available_index(exclude_current=True)
        if target is None:
            # exhausted: escalate so AgentSession can close
            self.emit("error", error)
            return

        # recoverable while a fallback remains, so the session isn't torn down
        self.emit("error", error.model_copy(update={"recoverable": True}))
        if self._swap_task is None or self._swap_task.done():
            # capture the speaking state now; the dead generation may flip it before the swap runs
            self._swap_task = asyncio.create_task(self._swap(target, self._is_agent_speaking()))
            self._swap_task.add_done_callback(self._observe_automatic_swap)

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
        """Replace the active child without holding state locks across external awaits."""
        async with self._swap_serialization_lock:
            completion = asyncio.get_running_loop().create_future()
            async with self._swap_state_lock:
                retiring_child = self._active
                self._swap_phase = "interrupting"
                self._swap_completion = completion

            candidate: RealtimeSession | None = None
            candidate_index: int | None = None
            try:
                # Interrupt outside the state lock. The retiring child stays current until the
                # interrupt has committed the heard content and a replacement is ready.
                if self._agent_session is not None:
                    try:
                        await self._agent_session.interrupt(force=True)
                    except Exception:
                        logger.debug("failed to interrupt the agent before swap", exc_info=True)

                async with self._swap_state_lock:
                    self._swap_phase = "replacing"

                # Text sync that intersected interruption has returned UNKNOWN by now, so its
                # locally committed message is part of the replay snapshot.
                if self._agent_session is not None:
                    chat_ctx = self._agent_session.current_agent.chat_ctx
                else:
                    chat_ctx = retiring_child.chat_ctx
                if self._provider_replay_excluded_item_ids:
                    chat_ctx = chat_ctx.copy()
                    replay_item_ids = {item.id for item in chat_ctx.items}
                    self._provider_replay_excluded_item_ids.intersection_update(replay_item_ids)
                    chat_ctx.items = [
                        item
                        for item in chat_ctx.items
                        if item.id not in self._provider_replay_excluded_item_ids
                    ]
                replayed_item_ids = frozenset(item.id for item in chat_ctx.items)

                async def _bring_up(
                    index: int,
                ) -> tuple[RealtimeSession | None, Exception | None]:
                    child = self._adapter._models[index].session(
                        turn_detection_disabled=self._turn_detection_disabled
                    )
                    try:
                        await child._update_session(
                            instructions=self._instructions,
                            chat_ctx=chat_ctx,
                            tools=self._tools,
                        )
                        if is_given(self._tool_choice):
                            child.update_options(tool_choice=self._tool_choice)
                        self._bind(child)
                        return child, None
                    except asyncio.CancelledError:
                        with contextlib.suppress(Exception):
                            await child.aclose()
                        raise
                    except Exception as error:
                        logger.exception("failed to start realtime model on swap, trying next")
                        with contextlib.suppress(Exception):
                            await child.aclose()
                        self._set_available(index, False)
                        return None, error

                self._unbind(retiring_child)
                with contextlib.suppress(Exception):
                    await retiring_child.aclose()

                error: Exception | None = None
                attempted_indices = {target_index}
                candidate, error = await _bring_up(target_index)
                candidate_index = target_index if candidate is not None else None
                while candidate is None:
                    next_index = self._next_available_index(excluded_indices=attempted_indices)
                    if next_index is None:
                        break
                    attempted_indices.add(next_index)
                    candidate, error = await _bring_up(next_index)
                    candidate_index = next_index if candidate is not None else None

                if candidate is None or candidate_index is None:
                    sync_error = RealtimeError(f"failed to replace realtime session: {error}")
                    async with self._swap_state_lock:
                        outcome = _SwapOutcome(
                            child_epoch=self._child_epoch,
                            replayed_item_ids=frozenset(),
                            error=sync_error,
                        )
                        self._swap_phase = "idle"
                        self._swap_completion = None
                        completion.set_result(outcome)
                    self.emit(
                        "error",
                        RealtimeModelError(
                            timestamp=time.time(),
                            label=self._adapter.label,
                            error=error or sync_error,
                            recoverable=False,
                        ),
                    )
                    return

                async with self._swap_state_lock:
                    self._active = candidate
                    self._active_index = candidate_index
                    self._child_epoch += 1
                    outcome = _SwapOutcome(
                        child_epoch=self._child_epoch,
                        replayed_item_ids=replayed_item_ids,
                    )
                    self._swap_phase = "idle"
                    self._swap_completion = None
                    completion.set_result(outcome)
                self._adapter._active_instance = self._adapter._models[candidate_index]

                self.emit("session_reconnected", RealtimeSessionReconnectedEvent())
                if (
                    was_speaking
                    and self._adapter._regenerate_on_swap
                    and self._agent_session is not None
                ):
                    self._agent_session.generate_reply()
            except BaseException as error:
                if candidate is not None and candidate is not self._active:
                    self._unbind(candidate)
                    with contextlib.suppress(Exception):
                        await candidate.aclose()
                sync_error = RealtimeError(f"realtime session swap did not complete: {error}")
                async with self._swap_state_lock:
                    self._swap_phase = "idle"
                    self._swap_completion = None
                    if not completion.done():
                        completion.set_result(
                            _SwapOutcome(
                                child_epoch=self._child_epoch,
                                replayed_item_ids=frozenset(),
                                error=sync_error,
                            )
                        )
                raise

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
        if self._swap_phase != "idle":
            # dropped; the swap replays the agent chat context afterwards
            return
        await self._active.update_chat_ctx(chat_ctx)
        # An explicit complete-context update makes every retained local item provider-visible.
        self._provider_replay_excluded_item_ids.clear()

    def _exclude_chat_ctx_item_from_replay(self, item_id: str) -> None:
        self._provider_replay_excluded_item_ids.add(item_id)

    async def _sync_user_message(
        self, chat_ctx: ChatContext, message_id: str
    ) -> _UserMessageSyncResult:
        while True:
            async with self._swap_state_lock:
                phase = self._swap_phase
                completion = self._swap_completion
                child = self._active
                child_epoch = self._child_epoch

            if phase == "interrupting":
                self._remember_cross_swap_generation(completion, chat_ctx, message_id)
                return _UserMessageSyncResult(
                    _UserMessageSyncStatus.UNKNOWN,
                    RealtimeError("user-message synchronization intersected a session swap"),
                )
            if phase == "replacing":
                assert completion is not None
                outcome = await asyncio.shield(completion)
                if outcome.error is not None:
                    return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, outcome.error)
                continue

            result = await child._sync_user_message(chat_ctx, message_id)
            async with self._swap_state_lock:
                child_still_current = (
                    self._swap_phase == "idle"
                    and self._active is child
                    and self._child_epoch == child_epoch
                )
                completion = self._swap_completion

            if not child_still_current:
                self._remember_cross_swap_generation(completion, chat_ctx, message_id)
                return _UserMessageSyncResult(
                    _UserMessageSyncStatus.UNKNOWN,
                    RealtimeError("user-message synchronization completed on a retired session"),
                )
            if result.status is not _UserMessageSyncStatus.REJECTED:
                self._provider_replay_excluded_item_ids.discard(message_id)
            return result

    def _remember_cross_swap_generation(
        self,
        completion: asyncio.Future[_SwapOutcome] | None,
        chat_ctx: ChatContext,
        message_id: str,
    ) -> None:
        task = asyncio.current_task()
        if task is None:
            return
        self._pending_cross_swap_generations[task] = _PendingCrossSwapGeneration(
            completion=completion,
            chat_ctx=chat_ctx,
            message_id=message_id,
        )
        task.add_done_callback(lambda done: self._pending_cross_swap_generations.pop(done, None))

    async def _generate_reply_after_swap(
        self,
        pending: _PendingCrossSwapGeneration,
        *,
        instructions: NotGivenOr[str],
        tool_choice: NotGivenOr[ToolChoice],
        tools: NotGivenOr[list[Tool]],
    ) -> GenerationCreatedEvent:
        completion = pending.completion
        last_outcome: _SwapOutcome | None = None
        while True:
            if completion is not None:
                last_outcome = await asyncio.shield(completion)
                completion = None
                if last_outcome.error is not None:
                    raise last_outcome.error

            async with self._swap_state_lock:
                if self._swap_phase != "idle":
                    completion = self._swap_completion
                    last_outcome = None
                    continue
                child = self._active
                child_epoch = self._child_epoch

            message_was_replayed = (
                last_outcome is not None
                and last_outcome.child_epoch == child_epoch
                and pending.message_id in last_outcome.replayed_item_ids
            )
            if not message_was_replayed:
                result = await child._sync_user_message(pending.chat_ctx, pending.message_id)
                async with self._swap_state_lock:
                    child_still_current = (
                        self._swap_phase == "idle"
                        and self._active is child
                        and self._child_epoch == child_epoch
                    )
                    if not child_still_current:
                        completion = self._swap_completion
                        last_outcome = None
                        continue
                if result.status is _UserMessageSyncStatus.REJECTED:
                    raise result.error or RealtimeError(
                        "replacement realtime session rejected the finalized user turn"
                    )
                self._provider_replay_excluded_item_ids.discard(pending.message_id)

            generation_fut = child.generate_reply(
                instructions=instructions,
                tool_choice=tool_choice,
                tools=tools,
            )
            return await generation_fut

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tools = tools
        await self._active.update_tools(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        self._tool_choice = tool_choice
        self._active.update_options(tool_choice=tool_choice)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._swap_phase != "idle":
            # drop during swap; replaying would lag the model
            return
        self._active.push_audio(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        if self._swap_phase != "idle":
            return
        self._active.push_video(frame)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        try:
            task = asyncio.current_task()
        except RuntimeError:  # synchronous provider setup has no running loop
            task = None
        pending = self._pending_cross_swap_generations.pop(task, None) if task is not None else None
        if pending is not None:
            return asyncio.create_task(
                self._generate_reply_after_swap(
                    pending,
                    instructions=instructions,
                    tool_choice=tool_choice,
                    tools=tools,
                ),
                name="FallbackRealtimeSession.generate_reply_after_swap",
            )
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
        # cancel an in-flight swap first, else its fresh child would leak past aclose
        if self._swap_task is not None:
            await aio.cancel_and_wait(self._swap_task)
        self._unbind(self._active)
        await self._active.aclose()
