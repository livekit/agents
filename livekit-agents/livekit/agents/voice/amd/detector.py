from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from ... import inference, llm, stt
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.participant import wait_for_participant_attribute, wait_for_track_publication
from . import _inference
from ._transcription import TurnTranscript
from .classifier import AMDCategory, AMDPredictionEvent
from .events import AMDCompletedEvent, AMDMenuObservedEvent

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..audio_recognition import _EndOfTurnInfo
    from ..events import AgentFalseInterruptionEvent, AgentStateChangedEvent, SpeechCreatedEvent
    from ..speech_handle import SpeechHandle

DEFAULT_SCREENING_INSTRUCTIONS = (
    "Call state: automated call screening. Answer the screening assistant's latest "
    "prompt briefly. Use what you know about who you are and why you are calling. "
    "Then wait for its next prompt."
)
DEFAULT_VOICEMAIL_INSTRUCTIONS = (
    "Call state: voicemail. Deliver one concise, self-contained message. "
    "State who you are, why you are calling, and the next step for the recipient."
)
DEFAULT_IVR_INSTRUCTIONS = (
    "Call state: an automated phone menu. Use the participant's actual prompt and "
    "the purpose of this call to choose the next step. Use send_dtmf_events for an "
    "explicit keypad choice, or give a short spoken answer when requested. "
    "Do not invent a menu option. After the action, wait for the next prompt."
)
_HUMAN_INSTRUCTIONS = (
    "Call state: a human has answered. The latest participant turn is from "
    "the human, not the automated system. Earlier automated prompts no longer "
    "apply. Respond to the human's latest message and continue the call normally. "
    "Do not resume a response to an earlier automated prompt."
)


@dataclass
class _Turn:
    turn_id: int
    committed_at: float
    transcript: str
    speech_duration: float
    decision: asyncio.Future[AMDPredictionEvent]
    dtmf_digits: str = ""
    timer: asyncio.TimerHandle | None = None
    timed_out: bool = False
    updated_turn_ids: set[int] = field(default_factory=set)


class AMD(EventEmitter[Literal["amd_prediction", "amd_completed", "amd_menu_observed"]]):
    """Experimental client-side, multi-turn answering-machine detection.

    Start an AgentSession before entering AMD. Enter AMD before creating a SIP
    participant. execute() waits for completion, not the first machine prediction.
    The SDK owns classification and stage control. Models can use any provider.

    This MVP uses the customer's current pipeline Agent. Realtime models and
    agent handoffs during AMD are not supported yet. The final active-Agent API
    remains undecided. Normal hooks, interruptions, playback, and StopResponse
    stay in AgentSession. Menu events are informational and never execute actions.

    Example:
        async with AMD(session) as amd:
            result = await amd.execute()

    Args:
        session: Started session whose participant audio and client-side EOT to use.
        llm: Classification and menu model. Defaults to the current Agent's LLM.
            A string selects a LiveKit Inference model. Supplied models stay open
            when AMD completes.
        stt: Optional second STT model. A string selects a LiveKit Inference model.
            None or NOT_GIVEN uses only the session transcript.
        participant_identity: Select the participant before placing an outbound call.
        wait_until_answered: Discard pre-answer audio from AMD and AgentSession
            when True. When False, listen to subscribed SIP early media.
        screening_instructions: Instructions for each screening turn.
        voicemail_instructions: Instructions for one message per voicemail stage.
        ivr_instructions: Instructions for each IVR turn. The SDK supplies a DTMF
            tool only to IVR reply generations.
        idle_timeout: Silent, inactive time before AMD completes outside voicemail.
        voicemail_idle_timeout: Silent, inactive time in voicemail. Starts after
            playback finishes, to allow a delayed post-message menu to arrive.
        timeout: Hard limit from the start of listening.
        inference_timeout: Prediction deadline per committed turn. A late result
            can update the stage, but cannot change a reply that already started.
        max_uncertain_turns: Consecutive uncertain predictions before completion.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        llm: NotGivenOr[llm.LLM | str] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | str | None] = NOT_GIVEN,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        wait_until_answered: bool = True,
        screening_instructions: str = DEFAULT_SCREENING_INSTRUCTIONS,
        voicemail_instructions: str = DEFAULT_VOICEMAIL_INSTRUCTIONS,
        ivr_instructions: str = DEFAULT_IVR_INSTRUCTIONS,
        idle_timeout: float = 10.0,
        voicemail_idle_timeout: float = 60.0,
        timeout: float = 120.0,
        inference_timeout: float = 1.5,
        max_uncertain_turns: int = 3,
    ) -> None:
        super().__init__()
        if (
            min(idle_timeout, voicemail_idle_timeout, timeout, inference_timeout) <= 0
            or max_uncertain_turns < 1
        ):
            raise ValueError("AMD timeouts and max_uncertain_turns must be positive")
        self._session = session
        self._owns_llm = isinstance(llm, str)
        self._llm = inference.LLM.from_model_string(llm) if isinstance(llm, str) else llm
        self._owns_stt = isinstance(stt, str)
        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt
        self._participant_identity = participant_identity
        self._wait_until_answered = wait_until_answered
        self._instructions = {
            AMDCategory.MACHINE_SCREENING: screening_instructions,
            AMDCategory.MACHINE_VM: voicemail_instructions,
            AMDCategory.MACHINE_IVR: ivr_instructions,
        }
        self._idle_timeout = idle_timeout
        self._voicemail_idle_timeout = voicemail_idle_timeout
        self._timeout = timeout
        self._inference_timeout = inference_timeout
        self._max_uncertain_turns = max_uncertain_turns
        self._session_id = uuid.uuid4().hex
        self._control_prefix = f"amd_{uuid.uuid4().hex}_"
        self._entered = False
        self._closed = False
        self._listening = False
        self._speaking_since: float | None = None
        self._speech_duration = 0.0
        self._category = AMDCategory.UNCERTAIN
        self._previous_turn: AMDCategory | None = None
        self._previous_stage: AMDCategory | None = None
        self._should_wait = False
        self._voicemail_message_played = False
        self._voicemail_started = False
        self._voicemail_handle: SpeechHandle | None = None
        self._voicemail_audio_start = 0
        self._uncertain_turns = 0
        self._inference_timeouts = 0
        self._turn_id = 0
        self._pending_dtmf_digits = ""
        self._turns: dict[int, _Turn] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=20)
        self._updated_turn_ids: set[int] = set()
        self._transcript: TurnTranscript | None = None
        self._transcripts: dict[int, TurnTranscript] = {}
        self._last_inference_turn_id = 0
        self._pending_turn: _Turn | None = None
        self._reused_turns: list[_Turn] = []
        self._classifier_task: asyncio.Task[None] | None = None
        self._menu_task: asyncio.Task[None] | None = None
        self._latest: AMDPredictionEvent | None = None
        self._completion: asyncio.Future[AMDCompletedEvent] | None = None
        self._finishing: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._hard_timer: asyncio.TimerHandle | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        self._speeches: set[SpeechHandle] = set()
        self._agent: Any = None

    @property
    def enabled(self) -> bool:
        return self._entered and not self._closed

    @property
    def pending(self) -> bool:
        return self.enabled

    @property
    def started(self) -> bool:
        return self.enabled and self._listening

    @property
    def _discard_pre_answer_audio(self) -> bool:
        return self.enabled and not self._listening

    async def __aenter__(self) -> AMD:
        if self._entered or self._completion is not None:
            raise RuntimeError("Use a new AMD instance for each run")
        activity = self._session._activity
        if activity is None:
            raise RuntimeError("Start AgentSession before entering AMD")
        if isinstance(activity.llm, llm.RealtimeModel):
            raise ValueError(
                "AMD MVP supports pipeline STT/LLM/TTS only; "
                "realtime reply control is not implemented yet"
            )
        if self._session.amd is not None:
            raise RuntimeError("AMD is already active")
        if self._session.options.ivr_detection or self._session._ivr_activity is not None:
            raise ValueError("Disable session-level ivr_detection when using AMD")
        if not is_given(self._llm):
            if not isinstance(activity.llm, llm.LLM):
                raise ValueError("AMD requires an LLM for classification")
            self._llm = activity.llm
        self._agent = activity._agent
        self._completion = asyncio.get_running_loop().create_future()
        self._transcript = self._new_transcript()
        self._entered = True
        self._session._amd = self
        activity._pause_authorization()
        self._session.on("speech_created", self._on_speech_created)
        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("agent_false_interruption", self._on_false_interruption)
        if self._session._room_io:
            self._session._room_io.room.on("participant_disconnected", self._on_disconnected)
            if is_given(self._participant_identity):
                self._session._room_io.set_participant(self._participant_identity)
        self._spawn(self._setup_listening())
        return self

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("AMD task failed", exc_info=error)
            self._finish("inference_error")

    async def execute(self) -> AMDCompletedEvent:
        if not self._entered or self._completion is None:
            raise RuntimeError("Enter AMD before calling execute()")
        return await asyncio.shield(self._completion)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._entered:
            return
        self._finish("cancelled")
        assert self._finishing is not None
        await asyncio.shield(self._finishing)

    async def _setup_listening(self) -> None:
        try:
            if not self._session._room_io:
                self._start_listening()
                return
            room = self._session._room_io.room
            publication = await wait_for_track_publication(
                room=room,
                identity=self._participant_identity
                if is_given(self._participant_identity)
                else None,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
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
                self._finish("participant_missing")
                return
            self._participant_identity = publisher.identity
            if (
                self._wait_until_answered
                and publisher.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ):
                await wait_for_participant_attribute(
                    room, identity=publisher.identity, attribute="sip.callStatus", value="active"
                )
            if not self._closed:
                self._start_listening()
        except RuntimeError:
            self._finish("participant_missing")

    def _start_listening(self) -> None:
        if self._closed or self._listening:
            return
        self._listening = True
        loop = asyncio.get_running_loop()
        self._hard_timer = loop.call_later(self._timeout, self._finish, "timeout")
        self._rearm_idle()
        logger.info("AMD listening", extra={"session_id": self._session_id})

    def _on_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._participant_identity:
            self._finish("participant_disconnected")

    def _new_transcript(self) -> TurnTranscript:
        return TurnTranscript(
            self._stt if is_given(self._stt) else None,
            self._session.conn_options.stt_conn_options,
            self._on_transcript_update,
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.started and self._transcript is not None:
            self._transcript.push_audio(frame)

    def _on_transcript_update(self, transcript: TurnTranscript) -> None:
        if self._closed:
            return
        for entry in self._history:
            if entry["turn_id"] == transcript.turn_id:
                updated = {**entry, **transcript.history()}
                if updated != entry:
                    entry.update(updated)
                    assert transcript.turn_id is not None
                    self._updated_turn_ids.add(transcript.turn_id)
                return

    def notify_dtmf_sent(self, digits: str) -> None:
        """Report digits after their local send succeeds, in send order.

        The built-in send_dtmf_events tool calls this automatically. Custom
        senders must call it after publish_dtmf returns successfully. Report
        each digit separately when a sequence can fail or be cancelled midway.
        The next client-side EOT includes these digits in classification context.
        This does not send DTMF to the participant or change the AMD stage.
        Calls outside an active AMD run are ignored.
        """
        if not self.enabled:
            return
        if not digits or any(digit not in "0123456789*#ABCD" for digit in digits):
            raise ValueError("digits must contain only 0-9, *, #, or A-D")
        self._pending_dtmf_digits += digits

    def _on_user_speech_started(self) -> None:
        if self.started:
            self._speaking_since = time.monotonic()
            self._cancel_idle()
            if activity := self._session._activity:
                activity._pause_authorization()

    def _on_user_speech_ended(self, silence_duration: float) -> None:
        if self._speaking_since is not None:
            self._speech_duration += max(
                0, time.monotonic() - self._speaking_since - silence_duration
            )
        self._speaking_since = None

    def _on_transcript(self, text: str) -> None:
        if self.started and self._transcript is not None:
            self._transcript.add_session_text(text)

    def _on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        if not self.started:
            return False
        self._turn_id += 1
        info.amd_turn_id = self._turn_id
        self._cancel_idle()
        if activity := self._session._activity:
            activity._pause_authorization()
        speech_duration = self._speech_duration
        self._speech_duration = 0
        if self._speaking_since is not None:
            speech_duration += time.monotonic() - self._speaking_since
            self._speaking_since = time.monotonic()
        turn = _Turn(
            self._turn_id,
            time.monotonic(),
            info.new_transcript.strip()[-16000:],
            speech_duration,
            asyncio.get_running_loop().create_future(),
            dtmf_digits=self._pending_dtmf_digits,
        )
        self._turns[turn.turn_id] = turn
        assert self._transcript is not None
        transcript = self._transcript
        transcript.commit(turn.turn_id, info.new_transcript)
        self._transcripts[turn.turn_id] = transcript
        self._transcript = self._new_transcript()
        self._history.append({**transcript.history(), "dtmf_digits": turn.dtmf_digits})
        self._updated_turn_ids.intersection_update(entry["turn_id"] for entry in self._history)
        self._pending_dtmf_digits = ""
        if not transcript.ready.is_set() and transcript.pending:
            self._spawn(self._wait_for_transcript(turn, transcript))
        else:
            self._start_classification(turn)
        return False

    async def _wait_for_transcript(self, turn: _Turn, transcript: TurnTranscript) -> None:
        try:
            await asyncio.wait_for(transcript.ready.wait(), min(0.5, self._inference_timeout))
        except asyncio.TimeoutError:
            pass
        if not self._closed:
            self._start_classification(turn)

    def _start_classification(self, turn: _Turn) -> None:
        transcript = self._transcripts[turn.turn_id]
        turn.transcript = transcript.text
        history = [entry for entry in self._history if entry["turn_id"] < turn.turn_id]
        history_ids = {entry["turn_id"] for entry in history}
        updated_turn_ids = self._updated_turn_ids & history_ids
        if not turn.transcript and not updated_turn_ids:
            if self._pending_turn is not None and not self._pending_turn.decision.done():
                self._reused_turns.append(turn)
            else:
                self._fallback(turn, "reused")
            return
        if turn.turn_id < self._last_inference_turn_id:
            self._fallback(turn, "superseded")
            return
        if self._classifier_task is not None:
            self._classifier_task.cancel()
        if self._pending_turn is not None:
            self._updated_turn_ids.update(self._pending_turn.updated_turn_ids)
            self._fallback(self._pending_turn, "superseded")
            self._flush_reused_turns()
        if self._menu_task is not None:
            self._menu_task.cancel()
        turn.updated_turn_ids = self._updated_turn_ids & history_ids
        self._updated_turn_ids.difference_update(turn.updated_turn_ids | {turn.turn_id})
        self._last_inference_turn_id = turn.turn_id
        self._pending_turn = turn
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=_inference.CLASSIFY_PROMPT)
        entry = {**transcript.history(), "dtmf_digits": turn.dtmf_digits}
        entry.pop("alternative_transcript", None)
        chat_ctx.add_message(
            role="user",
            content=json.dumps(
                {
                    "stage": self._category.value,
                    "allowed_next_categories": sorted(_inference.ALLOWED[self._category]),
                    "earlier_turns": history,
                    "updated_turn_ids": sorted(turn.updated_turn_ids),
                    "speech_duration": turn.speech_duration,
                    **entry,
                }
            ),
        )
        self._classifier_task = self._spawn(self._classify(turn, chat_ctx))
        turn.timer = asyncio.get_running_loop().call_later(
            max(0, turn.committed_at + self._inference_timeout - time.monotonic()),
            self._on_prediction_timeout,
            turn,
        )

    async def _classify(self, turn: _Turn, chat_ctx: llm.ChatContext) -> None:
        assert is_given(self._llm)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(_inference.classify(self._llm, chat_ctx), 30)
            if self._closed or self._pending_turn is not turn:
                return
            category = (
                self._category if result.category == AMDCategory.UNCERTAIN else result.category
            )
            if category not in _inference.ALLOWED[self._category]:
                raise ValueError("Invalid AMD stage transition")
        except Exception as exc:
            if not self._closed and self._pending_turn is turn:
                logger.warning(
                    "AMD classification failed",
                    extra={"turn_id": turn.turn_id, "error_type": type(exc).__name__},
                )
                self._fallback(turn, "inference_error")
                self._pending_turn = None
                self._flush_reused_turns()
            return
        if turn.timer:
            turn.timer.cancel()
        logger.debug(
            "AMD classification",
            extra={"turn_id": turn.turn_id, "raw_category": result.category.value},
        )
        changed = self._category != category
        event = AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=category,
            reason="late_prediction" if turn.timed_out else "prediction",
            transcript=turn.transcript,
            speech_duration=turn.speech_duration,
            delay=time.monotonic() - turn.committed_at,
            prev_turn_category=self._latest.category if self._latest else None,
            prev_stage_category=self._category if changed else self._previous_stage,
            state_changed=changed,
            inference_duration=time.monotonic() - started,
            voicemail_message_played=self._voicemail_message_played,
        )
        self._category = category
        self._previous_turn = event.prev_turn_category
        self._previous_stage = event.prev_stage_category
        self._should_wait = event.should_wait
        self._latest = event
        self._pending_turn = None
        if changed:
            self._cancel_idle()
            self._voicemail_started = False
            if activity := self._session._activity:
                activity._cancel_preemptive_generation()
        self._inference_timeouts = 0
        self._uncertain_turns = (
            self._uncertain_turns + 1 if category == AMDCategory.UNCERTAIN else 0
        )
        self._emit_prediction(event)
        if not turn.decision.done():
            turn.decision.set_result(event)
        if category in _inference.TERMINAL:
            self._finish("finished")
            return
        self._flush_reused_turns()
        if self._uncertain_turns >= self._max_uncertain_turns:
            self._finish("max_uncertain_turns")
        else:
            self._rearm_idle()
            if category == AMDCategory.MACHINE_IVR:
                self._menu_task = self._spawn(self._extract_menu(turn))

    def _flush_reused_turns(self) -> None:
        turns, self._reused_turns = self._reused_turns, []
        for turn in turns:
            self._fallback(turn, "reused")

    def _emit_prediction(self, event: AMDPredictionEvent) -> None:
        self.emit("amd_prediction", event)
        if (host := self._session._session_host) is not None:
            host._on_amd_prediction(event)
        logger.info(
            "AMD prediction",
            extra={
                "turn_id": event.turn_id,
                "category": event.category.value,
                "reason": event.reason,
                "session_id": self._session_id,
            },
        )

    def _fallback(self, turn: _Turn, reason: str) -> None:
        if turn.timer:
            turn.timer.cancel()
        if turn.decision.done():
            return
        event = AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=self._category,
            reason=reason,
            transcript=turn.transcript,
            speech_duration=turn.speech_duration,
            delay=time.monotonic() - turn.committed_at,
            prev_turn_category=self._previous_turn,
            prev_stage_category=self._previous_stage,
            should_wait=self._should_wait,
            voicemail_message_played=self._voicemail_message_played,
        )
        self._emit_prediction(event)
        self._latest = event
        turn.decision.set_result(event)
        self._rearm_idle()

    def _on_prediction_timeout(self, turn: _Turn) -> None:
        if self._closed or turn.decision.done():
            return
        turn.timed_out = True
        self._inference_timeouts += 1
        self._fallback(turn, "inference_timeout")
        self._flush_reused_turns()
        if self._inference_timeouts >= 3:
            self._finish("inference_timeout")

    async def _extract_menu(self, turn: _Turn) -> None:
        assert is_given(self._llm)
        started = time.monotonic()
        try:
            menu = await asyncio.wait_for(_inference.extract_menu(self._llm, turn.transcript), 5)
        except Exception as exc:
            logger.debug("AMD menu extraction failed", extra={"error_type": type(exc).__name__})
            return
        if self._closed or not (menu.menu or menu.options):
            return
        self.emit(
            "amd_menu_observed",
            AMDMenuObservedEvent(
                session_id=self._session_id,
                turn_id=turn.turn_id,
                menu=menu.menu,
                options=menu.options,
                extraction_duration=time.monotonic() - started,
            ),
        )

    async def _prepare_reply(self, info: _EndOfTurnInfo, chat_ctx: llm.ChatContext) -> bool:
        turn = self._turns.get(info.amd_turn_id or 0)
        if turn is None:
            return not self._closed or self._category != AMDCategory.MACHINE_UNAVAILABLE
        await asyncio.shield(turn.decision)
        if turn.turn_id != self._turn_id:
            return False
        if self._closed:
            if self._category == AMDCategory.HUMAN:
                chat_ctx.add_message(
                    id=f"{self._control_prefix}{turn.turn_id}",
                    role="user",
                    content=_HUMAN_INSTRUCTIONS,
                    extra={"amd_run": self._session_id, "amd_stage": self._category.value},
                )
            return self._category != AMDCategory.MACHINE_UNAVAILABLE
        if self._session.current_agent is not self._agent:
            self._finish("agent_changed")
            return False
        if self._should_wait:
            self._rearm_idle()
            return False
        if self._category == AMDCategory.MACHINE_VM and self._voicemail_started:
            self._rearm_idle()
            return False
        if self._category == AMDCategory.MACHINE_VM:
            self._voicemail_started = True
        if instructions := self._instructions.get(self._category):
            chat_ctx.add_message(
                id=f"{self._control_prefix}{turn.turn_id}",
                role="user",
                content=instructions
                + (
                    "\nThe voicemail message already played locally."
                    if self._category == AMDCategory.MACHINE_IVR and self._voicemail_message_played
                    else ""
                ),
                extra={"amd_run": self._session_id, "amd_stage": self._category.value},
            )
        if activity := self._session._activity:
            activity._resume_authorization()
        return True

    def _reply_tools(self, tools: list[llm.Tool | llm.Toolset]) -> list[llm.Tool | llm.Toolset]:
        if self.started and self._category == AMDCategory.MACHINE_IVR and not self._should_wait:
            from ...beta.tools.send_dtmf import send_dtmf_events

            if not any(tool.id == send_dtmf_events.id for tool in tools):
                return [*tools, send_dtmf_events]
        return tools

    def _on_reply_created(self, handle: SpeechHandle, turn_id: int | None) -> None:
        if (
            not self._closed
            and turn_id == self._turn_id
            and self._category == AMDCategory.MACHINE_VM
        ):
            self._voicemail_handle = handle
            output = self._session.output.audio
            self._voicemail_audio_start = output.captured_playout_segments if output else 0

    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        if self._closed:
            return
        self._cancel_idle()
        self._speeches.add(event.speech_handle)
        event.speech_handle.add_done_callback(self._on_speech_done)

    def _on_speech_done(self, handle: SpeechHandle) -> None:
        self._speeches.discard(handle)
        if (
            handle is self._voicemail_handle
            and not handle.interrupted
            and handle.exception() is None
        ):
            output = self._session.output.audio
            if output and output.captured_playout_segments > self._voicemail_audio_start:
                self._voicemail_message_played = True
        self._rearm_idle()

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        self._rearm_idle()

    def _on_false_interruption(self, event: AgentFalseInterruptionEvent) -> None:
        # AgentSession clears its pause state after it emits this event.
        asyncio.get_running_loop().call_soon(self._rearm_idle)

    def _cancel_idle(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _rearm_idle(self) -> None:
        activity = self._session._activity
        waiting = any(not turn.decision.done() for turn in self._turns.values())
        if (
            not self.started
            or self._speaking_since is not None
            or self._should_wait
            or waiting
            or self._speeches
            or activity is None
            or not activity._no_pending_speech
            or activity._paused_speech is not None
            or activity._false_interruption_timer is not None
            or activity._false_interruption_pending
            or self._session.agent_state in {"speaking", "thinking"}
            or (
                self._session.output.audio is not None
                and self._session.output.audio._pending_playback_count > 0
            )
            or (
                activity._user_turn_completed_atask is not None
                and not activity._user_turn_completed_atask.done()
            )
        ):
            self._cancel_idle()
            return
        if self._idle_timer is None:
            idle_timeout = (
                self._voicemail_idle_timeout
                if self._category == AMDCategory.MACHINE_VM
                else self._idle_timeout
            )
            self._idle_timer = asyncio.get_running_loop().call_later(
                idle_timeout, self._finish, "idle_timeout"
            )

    def _finish(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_idle()
        self._pending_dtmf_digits = ""
        if self._hard_timer:
            self._hard_timer.cancel()
        for turn in self._turns.values():
            if turn.timer:
                turn.timer.cancel()
            if not turn.decision.done():
                turn.decision.set_result(
                    self._latest
                    or AMDPredictionEvent(
                        turn_id=turn.turn_id,
                        category=self._category,
                        reason=reason,
                        transcript=turn.transcript,
                        speech_duration=turn.speech_duration,
                        delay=0,
                    )
                )
        self._finishing = asyncio.create_task(self._cleanup(reason))

    async def _cleanup(self, reason: str) -> None:
        try:
            await aio.cancel_and_wait(*self._tasks)
            transcripts = list(self._transcripts.values())
            if self._transcript is not None:
                transcripts.append(self._transcript)
            close_tasks = [transcript.aclose() for transcript in transcripts]
            if self._owns_stt and is_given(self._stt) and self._stt is not None:
                close_tasks.append(self._stt.aclose())
            if self._owns_llm and is_given(self._llm):
                close_tasks.append(self._llm.aclose())
            for error in await asyncio.gather(*close_tasks, return_exceptions=True):
                if isinstance(error, BaseException):
                    logger.warning(
                        "AMD resource cleanup failed", extra={"error_type": type(error).__name__}
                    )
            self._session.off("speech_created", self._on_speech_created)
            self._session.off("agent_state_changed", self._on_agent_state_changed)
            self._session.off("agent_false_interruption", self._on_false_interruption)
            if self._session._room_io:
                self._session._room_io.room.off("participant_disconnected", self._on_disconnected)
            for speech in self._speeches:
                speech.remove_done_callback(self._on_speech_done)
            if activity := self._session._activity:
                activity._cancel_preemptive_generation()
                # Cancel only queued/held replies. AgentSession owns current playback.
                if self._category == AMDCategory.MACHINE_UNAVAILABLE:
                    for _, _, speech in activity._speech_q:
                        speech._cancel()
                    current = activity._current_speech
                    if (
                        current
                        and self._session.agent_state != "speaking"
                        and activity._paused_speech is None
                    ):
                        current._cancel()
                activity._resume_authorization()
            if self._session._amd is self:
                self._session._amd = None
        finally:
            assert self._completion is not None
            result = AMDCompletedEvent(
                category=self._category,
                reason=reason,
                turn_id=self._latest.turn_id if self._latest else self._turn_id,
                transcript=self._latest.transcript if self._latest else "",
                prev_turn_category=self._previous_turn,
                prev_stage_category=self._previous_stage,
                voicemail_message_played=self._voicemail_message_played,
            )
            self._completion.set_result(result)
            self.emit("amd_completed", result)
