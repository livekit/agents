from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from ... import inference, llm, stt
from ..._exceptions import APIError
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.misc import is_cloud
from ...utils.participant import wait_for_participant_attribute, wait_for_track_publication
from . import _fsm, _inference
from ._chat_context import AMDChatContext, AMDRequest, Turn
from ._stt import AMDRacingSTT
from .events import (
    AMDCategory,
    AMDCompletedEvent,
    AMDLifecycle,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    AMDReason,
)

if TYPE_CHECKING:
    from .._turn_hooks import TurnHooks
    from ..agent import Agent
    from ..agent_session import AgentSession
    from ..events import (
        AgentFalseInterruptionEvent,
        AgentStateChangedEvent,
        FunctionToolsExecutedEvent,
        SpeechCreatedEvent,
        UserInputTranscribedEvent,
        UserStateChangedEvent,
    )
    from ..speech_handle import SpeechHandle

_DEFAULT_MAX_INFERENCE_TIMEOUTS = 3  # Allow transient delays before abandoning detection.
_MACHINE_SILENCE_THRESHOLD = 1.5  # Wait for a pause before replying to machines.
_MENU_TIMEOUT = 5.0  # Stop optional menu work before it becomes stale.
_TRACK_PUBLICATION_TIMEOUT = 5.0
_MACHINE_CATEGORIES = frozenset(
    {
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    }
)

_DEFAULT_SCREENING_INSTRUCTIONS = (
    "Call state: automated call screening. Answer the screening assistant's latest "
    "prompt briefly. Use what you know about who you are and why you are calling. "
    "Then wait for its next prompt."
)
_DEFAULT_VOICEMAIL_INSTRUCTIONS = (
    "Call state: voicemail. Deliver one concise, self-contained message. "
    "State who you are, why you are calling, and the next step for the recipient."
)
_DEFAULT_IVR_INSTRUCTIONS = (
    "Call state: an automated phone menu. Use the participant's actual prompt and "
    "the purpose of this call to choose the next step. Use send_dtmf_events for an "
    "explicit keypad choice, or give a short spoken answer when requested. "
    "Do not invent a menu option. After the action, wait for the next prompt."
)
_DEFAULT_HUMAN_INSTRUCTIONS = (
    "Call state: a human has answered. The latest participant turn is from "
    "the human, not the automated system. Earlier automated prompts no longer "
    "apply. Respond to the human's latest message and continue the call normally. "
    "Do not resume a response to an earlier automated prompt."
)


@dataclass(frozen=True)
class AMDOptions:
    idle_timeout: float
    voicemail_idle_timeout: float
    timeout: float
    inference_timeout: float
    machine_silence_threshold: float
    max_uncertain_turns: int
    max_inference_timeouts: int


@dataclass(frozen=True)
class ReplyDecision:
    allow: bool
    instructions_for: AMDCategory | None = None
    track_voicemail: bool = False


@dataclass
class _AMDResources:
    """Resources and saved settings from ``__aenter__`` until completion."""

    agent: Agent
    llm: llm.LLM
    completion: asyncio.Future[AMDCompletedEvent]
    stt: AMDRacingSTT
    session_allow_interruptions: bool
    agent_allow_interruptions: NotGivenOr[bool]


@dataclass
class _AMDTurnHooks:
    _amd: AMD
    _turn_id: int | None = None
    _track_voicemail: bool = False
    reply_instructions: str | None = None

    def on_user_turn_committed(self, transcript: str, end_of_turn_delay: float | None) -> TurnHooks:
        return self._amd._on_user_turn_committed(transcript, end_of_turn_delay)

    def on_user_turn_completed(self) -> None:
        self._amd._update_idle()

    def on_reply_generation(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        return self._amd._on_reply_generation(tools)

    async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
        if self._turn_id is None:
            return (
                self._amd._state is not AMDCategory.MACHINE_UNAVAILABLE
                and not self._amd._should_wait
            )
        decision = await self._amd._should_reply(self._turn_id, chat_ctx)
        self._track_voicemail = decision.track_voicemail
        if decision.instructions_for is not None:
            self.reply_instructions = self._amd._get_instructions(decision.instructions_for)
        return decision.allow

    def on_agent_turn_committed(self, handle: SpeechHandle) -> None:
        if self._track_voicemail:
            assert self._turn_id is not None
            self._amd._track_voicemail(self._turn_id, handle)


class AMD(EventEmitter[Literal["amd_prediction", "amd_completed", "amd_menu_observed"]]):
    """Multi-turn answering-machine detection.

    Start an AgentSession before entering AMD. Enter AMD before creating a SIP
    participant. execute() waits for completion, not the first machine prediction.
    The SDK owns classification and stage control. Models can use any provider.

    AMD uses the customer's current Agent. Realtime models require client-side
    turn detection, session or AMD STT, and per-response tool
    control. Agent handoffs during AMD are not supported. Normal hooks, interruptions,
    playback, and StopResponse stay in AgentSession. Menu events are informational
    and never execute actions.
    AMD enables interruptions on the session and current Agent, then restores
    their settings when detection finishes. Direct speech generation during AMD
    is not supported.

    Example:
        async with AMD(session) as amd:
            result = await amd.execute()

    Args:
        session: Started session whose participant audio and client-side EOT to use.
        llm: Classification and menu model. When omitted, use
            ``google/gemini-3.1-flash-lite`` if LiveKit Cloud credentials are
            available; otherwise use the current Agent's LLM. Pass None to
            always use the current Agent's LLM. A string selects a LiveKit
            Inference model. Calls use ``session.conn_options.llm_conn_options``.
            Supplied models stay open when AMD completes.
        stt: Optional streaming STT model for AMD. When omitted, use
            ``cartesia/ink-whisper`` if LiveKit Cloud credentials are available;
            otherwise use only the session transcript. Pass None to always use
            only the session transcript. AMD races this STT against the session
            transcript only when session STT is configured. A string selects a
            LiveKit Inference model.
        participant_identity: Select the participant before placing an outbound call.
        wait_until_answered: Discard pre-answer audio from AMD and AgentSession
            when True. When False, listen to subscribed SIP early media.
        screening_instructions: Instructions for each screening turn.
        voicemail_instructions: Instructions for one successfully played voicemail message.
        ivr_instructions: Instructions for each IVR turn. The SDK supplies a DTMF
            tool only to IVR reply generations.
        human_instructions: Temporary instructions for the first human turn after
            a machine stage. Skipped if no machine stage preceded the human.
        idle_timeout: Silent, inactive time before AMD completes outside voicemail.
        voicemail_idle_timeout: Silent, inactive time in voicemail. Starts after
            playback finishes, to allow a delayed post-message menu to arrive.
        timeout: Hard limit from the start of listening.
        inference_timeout: Prediction deadline per committed turn. Late results are ignored.
        machine_silence_threshold: Continuous participant silence before a machine
            reply is authorized. Includes silence before and during classification.
            Predictions update the category immediately. Replies outside a machine
            stage do not wait. Set to zero to disable the extra silence wait.
        max_uncertain_turns: Consecutive uncertain predictions before completion
            with the current stage. A wait prediction resets the count.
        max_inference_timeouts: Prediction timeouts before completion. A valid
            prediction resets the count.
    """

    _DEFAULT_LLM_MODEL: str = "google/gemini-3.1-flash-lite"
    _DEFAULT_STT_MODEL: str = "cartesia/ink-whisper"

    def __init__(
        self,
        session: AgentSession,
        *,
        llm: NotGivenOr[llm.LLM | str | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | str | None] = NOT_GIVEN,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        wait_until_answered: bool = True,
        screening_instructions: str = _DEFAULT_SCREENING_INSTRUCTIONS,
        voicemail_instructions: str = _DEFAULT_VOICEMAIL_INSTRUCTIONS,
        ivr_instructions: str = _DEFAULT_IVR_INSTRUCTIONS,
        human_instructions: str = _DEFAULT_HUMAN_INSTRUCTIONS,
        idle_timeout: float = 10.0,
        voicemail_idle_timeout: float = 60.0,
        timeout: float = 120.0,
        inference_timeout: float = 1.5,
        machine_silence_threshold: float = _MACHINE_SILENCE_THRESHOLD,
        max_uncertain_turns: int = 3,
        max_inference_timeouts: int = _DEFAULT_MAX_INFERENCE_TIMEOUTS,
    ) -> None:
        super().__init__()
        if (
            min(idle_timeout, voicemail_idle_timeout, timeout, inference_timeout) <= 0
            or max_uncertain_turns < 1
        ):
            raise ValueError("amd timeouts and max_uncertain_turns must be positive")
        if max_inference_timeouts < 1:
            raise ValueError("max_inference_timeouts must be positive")
        if machine_silence_threshold < 0:
            raise ValueError("machine_silence_threshold must be non-negative")
        if not is_given(llm) or not is_given(stt):
            api_key = os.getenv("LIVEKIT_INFERENCE_API_KEY") or os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_INFERENCE_API_SECRET") or os.getenv(
                "LIVEKIT_API_SECRET"
            )
            auto_select = (
                is_cloud(os.getenv("LIVEKIT_URL", "")) and bool(api_key) and bool(api_secret)
            )
            if not is_given(llm):
                llm = self._DEFAULT_LLM_MODEL if auto_select else None
            if not is_given(stt):
                stt = self._DEFAULT_STT_MODEL if auto_select else None

        self._session = session
        self._turn_hooks = _AMDTurnHooks(self)
        self._owns_llm = isinstance(llm, str)
        self._llm = inference.LLM.from_model_string(llm) if isinstance(llm, str) else llm
        self._owns_stt = isinstance(stt, str)
        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt
        if self._stt is not None and not self._stt.capabilities.streaming:
            raise ValueError("amd requires a streaming STT")
        self._participant_identity = participant_identity
        self._wait_until_answered = wait_until_answered
        self._instructions = {
            AMDCategory.MACHINE_SCREENING: screening_instructions,
            AMDCategory.MACHINE_VM: voicemail_instructions,
            AMDCategory.MACHINE_IVR: ivr_instructions,
            AMDCategory.HUMAN: human_instructions,
        }
        self._options = AMDOptions(
            idle_timeout=idle_timeout,
            voicemail_idle_timeout=voicemail_idle_timeout,
            timeout=timeout,
            inference_timeout=inference_timeout,
            machine_silence_threshold=machine_silence_threshold,
            max_uncertain_turns=max_uncertain_turns,
            max_inference_timeouts=max_inference_timeouts,
        )
        self._uncertain_turns = 0
        self._inference_timeouts = 0

        self._hard_deadline: float | None = None
        self._inference_deadline: float | None = None
        self._idle_deadline: float | None = None
        self._lifecycle = AMDLifecycle.INITIALIZED
        self._completion_reason = AMDReason.CANCELLED
        self._previous_stage: AMDCategory | None = None
        self._had_machine_stage = False
        self._state = AMDCategory.UNCERTAIN
        self._category = AMDCategory.UNCERTAIN  # latest accepted prediction, may be wait
        self._should_wait = False
        self._chat_ctx = AMDChatContext()
        self._turns: dict[int, Turn] = {}
        self._turn_id = 0
        self._started_at: float | None = None
        self._speech_started_at: float | None = None
        self._speech_ended_at: float | None = None
        self._speech_duration = 0.0
        self._speech_since_commit = False
        self._pending_turn: Turn | None = None
        self._latest: AMDPredictionEvent | None = None
        self._run: _AMDResources | None = None
        self._speeches: set[SpeechHandle] = set()

        self._session_id = uuid.uuid4().hex
        self._control_prefix = f"amd_{uuid.uuid4().hex}_"

        self._voicemail_turn_id: int | None = None
        self._voicemail_message_played = False
        self._voicemail_handle: SpeechHandle | None = None
        self._voicemail_audio_start = 0

        self._prediction_changed = asyncio.Event()
        self._classifier_atask: asyncio.Task[None] | None = None
        self._menu_atask: asyncio.Task[None] | None = None
        self._finish_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._subscriptions: list[tuple[EventEmitter[Any], str, Callable[..., Any]]] = []
        self._timer: asyncio.TimerHandle | None = None

    @property
    def _resources(self) -> _AMDResources:
        if self._run is None:
            raise RuntimeError("enter AMD before use")
        return self._run

    @property
    def lifecycle(self) -> AMDLifecycle:
        """Current lifecycle state of this AMD run."""
        return self._lifecycle

    @property
    def _blocks_session_audio(self) -> bool:
        return self.lifecycle not in {AMDLifecycle.ACTIVE, AMDLifecycle.FINISHED}

    async def __aenter__(self) -> AMD:
        activity = self._session._activity
        if activity is None:
            raise RuntimeError("start AgentSession before entering AMD")
        if isinstance(activity.llm, llm.RealtimeModel):
            if activity._rt_turn_detection_enabled:
                raise ValueError("amd requires client-side turn detection with realtime models")
            capabilities = activity.llm.capabilities
            if capabilities.auto_tool_reply_generation or not capabilities.per_response_tool_choice:
                raise ValueError(
                    "amd requires a realtime model with per-response tools and "
                    "client-controlled tool replies"
                )
            if activity.stt is None and self._stt is None:
                raise ValueError("amd requires session STT or AMD STT with realtime models")
        if self._session.amd:
            raise RuntimeError("amd is already active")
        if self._session.options.ivr_detection or self._session._ivr_activity is not None:
            raise ValueError("please disable session-level ivr_detection when using AMD")

        model = self._llm or activity.llm
        if not isinstance(model, llm.LLM):
            raise ValueError("amd requires an LLM for classification")

        if self.lifecycle is not AMDLifecycle.INITIALIZED:
            raise RuntimeError("use a new AMD instance for each run")

        self._lifecycle = AMDLifecycle.PENDING
        self._run = _AMDResources(
            agent=activity.agent,
            llm=model,
            completion=asyncio.get_running_loop().create_future(),
            stt=AMDRacingSTT(
                self._stt,
                self._session.conn_options.stt_conn_options,
                race_session=activity.stt is not None,
            ),
            session_allow_interruptions=self._session.options.interruption["enabled"],
            agent_allow_interruptions=activity.agent.allow_interruptions,
        )
        try:
            self._started_at = time.time()
            self._session._amd = self
            self._session._turn_hooks = self._turn_hooks
            self._session.options.interruption["enabled"] = True
            activity.agent._allow_interruptions = True
            activity._pause_authorization()

            self._subscriptions = [
                (self._session, "user_state_changed", self._on_user_state_changed),
                (self._session, "user_input_transcribed", self._on_user_input_transcribed),
                (self._session, "function_tools_executed", self._on_function_tools_executed),
                (self._session, "speech_created", self._on_speech_created),
                (self._session, "agent_state_changed", self._on_agent_state_changed),
                (self._session, "agent_false_interruption", self._on_false_interruption),
            ]
            if self._session._room_io:
                room = self._session._room_io.room
                self._subscriptions.append(
                    (room, "participant_disconnected", self._on_disconnected)
                )
                if is_given(self._participant_identity):
                    self._session._room_io.set_participant(self._participant_identity)

            for emitter, event, handler in self._subscriptions:
                emitter.on(event, handler)
            self._spawn(self._setup_listening())
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self.lifecycle is AMDLifecycle.INITIALIZED:
            return
        self._finish(AMDReason.CANCELLED)
        if self._finish_atask is not None:
            await asyncio.shield(self._finish_atask)

    async def execute(self) -> AMDCompletedEvent:
        return await asyncio.shield(self._resources.completion)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.lifecycle is AMDLifecycle.ACTIVE:
            self._resources.stt.push_audio(frame)

    async def _setup_listening(self) -> None:

        def _start_listening() -> None:
            if self.lifecycle is not AMDLifecycle.PENDING:
                return
            self._hard_deadline = time.monotonic() + self._options.timeout
            self._lifecycle = AMDLifecycle.ACTIVE
            if self._session.user_state == "speaking":
                self._speech_started_at = time.monotonic()
                self._speech_since_commit = True
            self._update_idle()
            logger.info("amd started listening", extra={"session_id": self._session_id})

        try:
            if not self._session._room_io:
                _start_listening()
                return
            room = self._session._room_io.room
            publication = await asyncio.wait_for(
                wait_for_track_publication(
                    room=room,
                    identity=self._participant_identity
                    if is_given(self._participant_identity)
                    else None,
                    kind=rtc.TrackKind.KIND_AUDIO,
                    wait_for_subscription=True,
                ),
                timeout=_TRACK_PUBLICATION_TIMEOUT,
            )
            publisher = next(
                (
                    p
                    for p in room.remote_participants.values()
                    if publication.sid in p.track_publications
                ),
                None,
            )
            if publisher is None:
                self._finish(AMDReason.PARTICIPANT_MISSING)
                return
            self._participant_identity = publisher.identity
            if (
                self._wait_until_answered
                and publisher.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ):
                await wait_for_participant_attribute(
                    room, identity=publisher.identity, attribute="sip.callStatus", value="active"
                )
            if self.lifecycle is not AMDLifecycle.FINISHED:
                _start_listening()
        except (RuntimeError, asyncio.TimeoutError):
            reason = AMDReason.PARTICIPANT_MISSING
            if (room_io := self._session._room_io) and not room_io.room.isconnected():
                reason = AMDReason.PARTICIPANT_DISCONNECTED
            self._finish(reason)

    # region: hooks

    def _on_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._participant_identity:
            self._finish(AMDReason.PARTICIPANT_DISCONNECTED)

    def _on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        if self.lifecycle not in {AMDLifecycle.PENDING, AMDLifecycle.ACTIVE}:
            return
        assert self._started_at is not None
        for call, output in event.zipped():
            if call.created_at >= self._started_at:
                self._chat_ctx.add_tool_result(call, output)

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            return
        now = time.monotonic()
        # Anchor the Unix speech boundary to the monotonic clock, preserving delayed edges.
        delay = (
            max(0, time.time() - event.speech_timestamp)
            if event.speech_timestamp is not None
            else 0
        )
        if event.new_state == "speaking":
            self._speech_started_at = now - delay
            self._speech_ended_at = None
            self._speech_since_commit = True
        elif event.old_state == "speaking":
            if self._speech_started_at is not None:
                self._speech_duration += max(0.0, now - delay - self._speech_started_at)
            self._speech_started_at = None
            self._speech_ended_at = now - delay
        self._prediction_changed.set()
        self._update_idle()

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        if self.lifecycle is AMDLifecycle.ACTIVE and event.is_final:
            self._resources.stt.push_transcript(event.transcript)

    def _on_user_turn_committed(
        self, transcript: str, end_of_turn_delay: float | None
    ) -> TurnHooks:
        if self.lifecycle is not AMDLifecycle.ACTIVE or self._check_hard_timeout():
            return self._turn_hooks
        if activity := self._session._activity:
            activity._pause_authorization()
        turn_transcript = self._resources.stt.end_turn(transcript)
        now = time.monotonic()
        speaking = self._session.user_state == "speaking"
        if not speaking and not self._speech_since_commit:
            # No speech edge since the last commit: use the EOT silence estimate.
            self._speech_ended_at = now - max(0.0, end_of_turn_delay or 0)
        duration, self._speech_duration = self._speech_duration, 0.0
        if speaking and self._speech_started_at is not None:
            duration += max(0.0, now - self._speech_started_at)
            self._speech_started_at = now
        self._speech_since_commit = False
        self._turn_id += 1
        turn = Turn(
            turn_id=self._turn_id,
            committed_at=now,
            transcript=turn_transcript,
            speech_duration=duration,
        )
        self._turns[turn.turn_id] = turn
        self._voicemail_turn_id = None
        if turn_transcript.transcript:
            self._chat_ctx.add_transcript(turn)
            self._cancel_classification()
            self._pending_turn = turn
            self._inference_deadline = now + self._options.inference_timeout
            request = self._chat_ctx.create_request(
                turn,
                stage=self._state,
                allowed=sorted(_fsm.ALLOWED[self._state]),
            )
            if self._menu_atask is not None:
                self._menu_atask.cancel()
                self._menu_atask = None
            self._classifier_atask = self._spawn(self._classify(turn, request))
        elif self._pending_turn is None:
            self._record_prediction(turn, AMDReason.REUSED)
        self._prediction_changed.set()
        self._update_idle()
        if (
            activity is not None
            and isinstance(activity.llm, llm.RealtimeModel)
            and not activity.llm.capabilities.user_transcription
            and activity.stt is None
            and turn_transcript.transcript
        ):
            message = llm.ChatMessage(
                role="user", content=[turn_transcript.transcript], transcript_confidence=0.0
            )
            activity.agent._chat_ctx.insert(message)
            self._session._conversation_item_added(message)
        return _AMDTurnHooks(self, turn.turn_id)

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        self._update_idle()

    def _on_false_interruption(self, event: AgentFalseInterruptionEvent) -> None:
        # AgentSession clears its pause state after it emits this event.
        asyncio.get_running_loop().call_soon(self._update_idle)

    # endregion

    # region: classification and extraction

    def _cancel_classification(self) -> None:
        task, self._classifier_atask = self._classifier_atask, None
        self._pending_turn = None
        self._inference_deadline = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _record_prediction(
        self,
        turn: Turn,
        reason: AMDReason,
        *,
        effects: tuple[_fsm.Effect, ...] = (),
        state_changed: bool = False,
    ) -> None:
        event = AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=self._category,
            stage=self._state,
            state_changed=state_changed,
            reason=reason,
            transcript=turn.transcript.transcript,
            speech_duration=turn.speech_duration,
            delay=time.monotonic() - turn.committed_at,
            inference_duration=turn.inference_duration if reason is AMDReason.PREDICTION else None,
            prev_turn_category=self._latest.category if self._latest else None,
            prev_stage_category=self._previous_stage,
            voicemail_message_played=self._voicemail_message_played,
        )
        turn.prediction = event
        if reason is not AMDReason.REUSED:
            # reused events repeat this one, so completion keeps the classified turn
            self._latest = event
        # Empty turns committed during inference reuse its result, each with its own event.
        # A non-empty commit would have cancelled this inference already.
        events = [event]
        for turn_id in range(turn.turn_id + 1, self._turn_id + 1):
            later = self._turns[turn_id]
            later.prediction = events[-1].model_copy(
                update={
                    "turn_id": later.turn_id,
                    "reason": AMDReason.REUSED,
                    "state_changed": False,
                    "transcript": later.transcript.transcript,
                    "speech_duration": later.speech_duration,
                    "delay": time.monotonic() - later.committed_at,
                    "inference_duration": None,
                    "prev_turn_category": events[-1].category,
                }
            )
            events.append(later.prediction)

        # Execute effects before notifying listeners, which may commit the next turn.
        for effect in effects:
            if effect is _fsm.Effect.COMPLETE:
                self._finish(AMDReason.FINISHED)
            elif effect is _fsm.Effect.EXTRACT_MENU:
                self._menu_atask = self._spawn(
                    self._extract_menu(turn.turn_id, turn.transcript.transcript)
                )
        if self._uncertain_turns >= self._options.max_uncertain_turns:
            self._finish(AMDReason.MAX_UNCERTAIN_TURNS)

        self._prediction_changed.set()
        self._update_idle()

        try:

            def _release_prediction(event: AMDPredictionEvent) -> None:
                if event.state_changed and (activity := self._session._activity):
                    activity._cancel_preemptive_generation()
                self.emit("amd_prediction", event)
                self._session._on_amd_prediction(event)
                logger.info(
                    "amd prediction",
                    extra={
                        "turn_id": event.turn_id,
                        "category": event.category.value,
                        "reason": event.reason,
                        "session_id": self._session_id,
                    },
                )

            for released in events:
                _release_prediction(released.model_copy())
        except Exception:
            logger.exception("amd prediction handler failed")
            self._finish(AMDReason.INTERNAL_ERROR)

    async def _classify(self, turn: Turn, request: AMDRequest) -> None:
        started = time.monotonic()
        category: AMDCategory | None = None
        try:
            result = await _inference.classify(
                self._resources.llm,
                request,
                conn_options=self._session.conn_options.llm_conn_options,
            )
        except Exception as exc:
            if asyncio.current_task() is not self._classifier_atask:
                return
            if not isinstance(exc, (APIError, ValueError, asyncio.TimeoutError)):
                raise
            logger.warning(
                "amd classification failed",
                extra={"turn_id": turn.turn_id, "error_type": type(exc).__name__},
            )
        else:
            turn.inference_duration = time.monotonic() - started
            category = result.category
        # Cancellation is best effort. Providers may return after timeout or supersession.
        if asyncio.current_task() is not self._classifier_atask or self._check_hard_timeout():
            return
        if self._inference_deadline is not None and time.monotonic() >= self._inference_deadline:
            self._timeout_inference()
            return
        self._cancel_classification()
        if category is None or category not in _fsm.ALLOWED[self._state]:
            self._record_prediction(turn, AMDReason.INFERENCE_ERROR)
        else:

            def _accept_prediction(turn: Turn, category: AMDCategory) -> None:
                result = _fsm.transition(self._state, category)
                state_changed = result.next_state != self._state
                if state_changed:
                    self._previous_stage = self._state
                    self._idle_deadline = None
                self._state = result.next_state
                self._category = category
                self._should_wait = category is AMDCategory.WAIT
                self._had_machine_stage |= self._state in _MACHINE_CATEGORIES
                self._inference_timeouts = 0
                self._uncertain_turns = (
                    self._uncertain_turns + 1 if category is AMDCategory.UNCERTAIN else 0
                )
                self._record_prediction(
                    turn,
                    AMDReason.PREDICTION,
                    effects=result.effects,
                    state_changed=state_changed,
                )

            _accept_prediction(turn, category)

    async def _extract_menu(self, turn_id: int, transcript: str) -> None:
        started = time.monotonic()
        try:
            menu = await asyncio.wait_for(
                _inference.extract_ivr_menu(
                    self._resources.llm,
                    transcript,
                    conn_options=self._session.conn_options.llm_conn_options,
                ),
                _MENU_TIMEOUT,
            )
        except Exception as exc:
            logger.debug("amd menu extraction failed", extra={"error_type": type(exc).__name__})
            return
        if (
            asyncio.current_task() is not self._menu_atask
            or self.lifecycle is AMDLifecycle.FINISHED
            or not (menu.menu or menu.options)
        ):
            return
        self.emit(
            "amd_menu_observed",
            AMDMenuObservedEvent(
                session_id=self._session_id,
                turn_id=turn_id,
                menu=menu.menu,
                options=menu.options,
                extraction_duration=time.monotonic() - started,
            ),
        )

    # endregion

    # region: reply controls and hooks

    async def _should_reply(self, turn_id: int, chat_ctx: llm.ChatContext) -> ReplyDecision:
        """Wait for the turn's prediction, then add stage instructions when a reply is allowed."""
        if turn_id in self._turns:
            # wait for the prediction for the given turn
            while (
                turn_id == self._turn_id
                and self.lifecycle is not AMDLifecycle.FINISHED
                and (
                    self._turns[turn_id].prediction is None or self._reply_held_at(time.monotonic())
                )
            ):
                self._prediction_changed.clear()
                await self._prediction_changed.wait()

            if (
                self.lifecycle is not AMDLifecycle.FINISHED
                and self._session.current_agent is not self._resources.agent
            ):
                self._finish(AMDReason.AGENT_CHANGED)
                return ReplyDecision(allow=False)
        if (
            self.lifecycle is AMDLifecycle.FINISHED
            and self._state is AMDCategory.MACHINE_UNAVAILABLE
        ):
            return ReplyDecision(allow=False)
        if turn_id not in self._turns:
            return ReplyDecision(allow=True)
        if turn_id != self._turn_id:
            return ReplyDecision(allow=False)
        if self._hard_deadline is not None and time.monotonic() >= self._hard_deadline:
            self._finish(AMDReason.TIMEOUT)
        reply_decision = self._authorize_reply(turn_id)
        if not reply_decision.allow:
            return reply_decision
        if reply_decision.instructions_for is not None:
            self._add_instructions(chat_ctx, reply_decision.instructions_for, turn_id)
        if (
            turn_id in self._turns
            and self.lifecycle is AMDLifecycle.ACTIVE
            and (activity := self._session._activity)
        ):
            activity._resume_authorization()
        return reply_decision

    def _authorize_reply(self, turn_id: int) -> ReplyDecision:
        category = self._state
        if self._should_wait:
            return ReplyDecision(allow=False)
        if self.lifecycle is AMDLifecycle.FINISHED:
            human_after_machine = category is AMDCategory.HUMAN and self._had_machine_stage
            return ReplyDecision(
                allow=category is not AMDCategory.MACHINE_UNAVAILABLE,
                instructions_for=AMDCategory.HUMAN if human_after_machine else None,
            )
        if self._turns[turn_id].prediction is None:
            return ReplyDecision(allow=False)
        if category is AMDCategory.MACHINE_VM:
            if (
                self._voicemail_message_played
                or self._voicemail_handle is not None
                or self._voicemail_turn_id is not None
            ):
                return ReplyDecision(allow=False)
            self._voicemail_turn_id = turn_id
            return ReplyDecision(allow=True, instructions_for=category, track_voicemail=True)
        return ReplyDecision(
            allow=True, instructions_for=None if category is AMDCategory.UNCERTAIN else category
        )

    def _get_instructions(self, stage: AMDCategory) -> str:
        content = self._instructions[stage]
        if stage is AMDCategory.MACHINE_IVR and self._voicemail_message_played:
            content += "\nThe voicemail message already played locally."
        return content

    def _add_instructions(
        self, chat_ctx: llm.ChatContext, stage: AMDCategory, turn_id: int
    ) -> None:
        chat_ctx.add_message(
            id=f"{self._control_prefix}{turn_id}",
            role="user",
            content=self._get_instructions(stage),
            extra={"amd_run": self._session_id, "amd_stage": self._state.value},
        )

    def _on_reply_generation(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        if self.lifecycle is AMDLifecycle.ACTIVE and self._state == AMDCategory.MACHINE_IVR:
            from ...beta.tools.send_dtmf import send_dtmf_events

            if not any(tool.id == send_dtmf_events.id for tool in tools):
                return [*tools, send_dtmf_events]
        return tools

    def _reply_held_at(self, now: float) -> bool:
        if (
            self.lifecycle is not AMDLifecycle.ACTIVE
            or not self._turn_id
            or self._options.machine_silence_threshold == 0
        ):
            return False
        prediction = self._turns[self._turn_id].prediction
        return (
            prediction is not None
            and not self._should_wait
            and self._state in _MACHINE_CATEGORIES
            and (
                self._session.user_state == "speaking"
                or self._speech_ended_at is None
                or now < self._speech_ended_at + self._options.machine_silence_threshold
            )
        )

    # endregion

    # region: track voicemail speech handle

    def _track_voicemail(self, turn_id: int, handle: SpeechHandle) -> None:
        if self.lifecycle is not AMDLifecycle.ACTIVE or self._voicemail_turn_id != turn_id:
            return
        self._voicemail_turn_id = None
        if self._voicemail_handle is not None:
            self._voicemail_handle.remove_done_callback(self._on_voicemail_done)
        self._voicemail_handle = handle
        output = self._session.output.audio
        self._voicemail_audio_start = output.captured_playout_segments if output else 0
        handle.add_done_callback(self._on_voicemail_done)

    def _on_voicemail_done(self, handle: SpeechHandle) -> None:
        if handle is not self._voicemail_handle:
            return
        handle.remove_done_callback(self._on_voicemail_done)
        self._voicemail_handle = None
        if (
            self.lifecycle is AMDLifecycle.FINISHED
            or handle.interrupted
            or handle.exception() is not None
        ):
            return
        output = self._session.output.audio
        if output and output.captured_playout_segments > self._voicemail_audio_start:
            self._voicemail_message_played = True
            if self._state is AMDCategory.MACHINE_VM:
                self._idle_deadline = None
        self._update_idle()

    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        if self.lifecycle is AMDLifecycle.FINISHED:
            return
        self._speeches.add(event.speech_handle)
        event.speech_handle.add_done_callback(self._on_speech_done)
        self._update_idle()

    def _on_speech_done(self, handle: SpeechHandle) -> None:
        self._speeches.discard(handle)
        self._update_idle()

    # endregion

    # region: timeouts and deadlines

    def _check_hard_timeout(self) -> bool:
        if self._hard_deadline is not None and time.monotonic() >= self._hard_deadline:
            self._finish(AMDReason.TIMEOUT)
            return True
        return False

    def _timeout_inference(self) -> None:
        turn = self._pending_turn
        self._cancel_classification()
        if turn is not None:
            self._inference_timeouts += 1
            self._record_prediction(turn, AMDReason.INFERENCE_TIMEOUT)

    def _update_idle(self) -> None:
        if self.lifecycle is not AMDLifecycle.ACTIVE or self._check_hard_timeout():
            return
        now = time.monotonic()
        reply_held = self._reply_held_at(now)
        if (
            self._inference_timeouts >= self._options.max_inference_timeouts
            and self._pending_turn is None
            and not reply_held
        ):
            self._finish(AMDReason.INFERENCE_TIMEOUT)
            return
        activity = self._session._activity
        busy = (
            self._pending_turn is not None
            or self._should_wait  # hold music or an advertisement; the hard timeout still applies
            or reply_held
            or self._session.user_state == "speaking"
            or bool(self._speeches)
            or activity is None
            or activity._is_agent_busy
        )
        if busy:
            self._idle_deadline = None
        elif self._idle_deadline is None:
            timeout = (
                self._options.voicemail_idle_timeout
                if self._state is AMDCategory.MACHINE_VM
                else self._options.idle_timeout
            )
            self._idle_deadline = now + timeout
        self._arm_timer(now=now)

    def _deadline_at(self, now: float) -> float | None:
        silence_deadline = None
        if self._reply_held_at(now) and self._speech_ended_at is not None:
            silence_deadline = self._speech_ended_at + self._options.machine_silence_threshold

        return min(
            (
                at
                for at in (
                    self._hard_deadline,
                    self._inference_deadline,
                    self._idle_deadline,
                    silence_deadline,
                )
                if at is not None
            ),
            default=None,
        )

    def _arm_timer(self, *, now: float | None = None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if (at := self._deadline_at(now if now is not None else time.monotonic())) is not None:

            def _on_deadline() -> None:
                if self.lifecycle is not AMDLifecycle.ACTIVE or self._check_hard_timeout():
                    return
                now = time.monotonic()
                if self._inference_deadline is not None and now >= self._inference_deadline:
                    self._timeout_inference()
                elif self._idle_deadline is not None and now >= self._idle_deadline:
                    self._finish(AMDReason.IDLE_TIMEOUT)
                self._prediction_changed.set()
                self._update_idle()

            self._timer = asyncio.get_running_loop().call_later(
                max(0, at - time.monotonic()), _on_deadline
            )

    # endregion

    # region: settle and clean up

    def _finish(self, reason: AMDReason) -> None:
        if self.lifecycle is AMDLifecycle.FINISHED:
            return
        self._lifecycle = AMDLifecycle.FINISHED
        self._completion_reason = reason
        self._hard_deadline = None
        self._idle_deadline = None
        self._cancel_classification()
        self._prediction_changed.set()
        self._arm_timer()
        for emitter, event, handler in self._subscriptions:
            emitter.off(event, handler)
        for speech in self._speeches:
            speech.remove_done_callback(self._on_speech_done)
        if self._voicemail_handle is not None:
            self._voicemail_handle.remove_done_callback(self._on_voicemail_done)
            self._voicemail_handle = None
        if self._session._amd is self:
            self._session.options.interruption["enabled"] = (
                self._resources.session_allow_interruptions
            )
            self._resources.agent._allow_interruptions = self._resources.agent_allow_interruptions
            self._session._amd = None
            if self._session._turn_hooks is self._turn_hooks:
                self._session._turn_hooks = None
            if activity := self._session._activity:
                if self._state == AMDCategory.MACHINE_UNAVAILABLE:
                    activity._cancel_pending_speeches()
                else:
                    activity._cancel_preemptive_generation()
                activity._resume_authorization()
        self._finish_atask = asyncio.create_task(self._cleanup())

    def _completion(self) -> AMDCompletedEvent:
        if self.lifecycle is not AMDLifecycle.FINISHED:
            raise RuntimeError("AMD has not completed")
        return AMDCompletedEvent(
            category=self._state,
            reason=self._completion_reason,
            turn_id=self._latest.turn_id if self._latest else self._turn_id,
            transcript=self._latest.transcript if self._latest else "",
            prev_turn_category=self._latest.prev_turn_category if self._latest else None,
            prev_stage_category=self._previous_stage,
            voicemail_message_played=self._voicemail_message_played,
        )

    async def _cleanup(self) -> None:
        try:
            await aio.cancel_and_wait(*self._tasks)
            close_tasks = [self._resources.stt.aclose()]
            if self._owns_stt and self._stt is not None:
                close_tasks.append(self._stt.aclose())
            if self._owns_llm and self._llm is not None:
                close_tasks.append(self._llm.aclose())
            for error in await asyncio.gather(*close_tasks, return_exceptions=True):
                if isinstance(error, BaseException):
                    logger.warning(
                        "amd resource cleanup failed", extra={"error_type": type(error).__name__}
                    )
        finally:
            result = self._completion()
            self._resources.completion.set_result(result)
            self.emit("amd_completed", result)

    # endregion

    # region: utilities

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run a background task that cleanup cancels and that ends the run if it fails."""
        try:
            task = asyncio.create_task(coro)
        except BaseException:
            coro.close()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("amd task failed", exc_info=error)
            self._finish(AMDReason.INTERNAL_ERROR)

    # endregion
