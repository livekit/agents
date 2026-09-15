from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from ... import inference, llm, stt
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.misc import is_cloud
from ...utils.participant import wait_for_participant_attribute, wait_for_track_publication
from . import _inference
from ._fsm import _AMDFSM, ClassifyRequest, _AMDEvent
from ._transcription import TurnTranscript
from .events import (
    AMDCategory,
    AMDCompletedEvent,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    AMDReason,
)

if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_session import AgentSession
    from ..audio_recognition import _EndOfTurnInfo
    from ..events import AgentFalseInterruptionEvent, AgentStateChangedEvent, SpeechCreatedEvent
    from ..speech_handle import SpeechHandle

MACHINE_SILENCE_THRESHOLD = 1.5

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
class _Run:
    """Resources that exist from ``__aenter__`` until completion."""

    agent: Agent
    llm: llm.LLM
    completion: asyncio.Future[AMDCompletedEvent]
    transcript: TurnTranscript
    """Transcript of the open turn. Replaced at each client-side EOT."""


class AMD(EventEmitter[Literal["amd_prediction", "amd_completed", "amd_menu_observed"]]):
    """Client-side, multi-turn answering-machine detection.

    Start an AgentSession before entering AMD. Enter AMD before creating a SIP
    participant. execute() waits for completion, not the first machine prediction.
    The SDK owns classification and stage control. Models can use any provider.

    AMD uses the customer's current pipeline Agent. Realtime models and
    agent handoffs during AMD are not supported. Normal hooks, interruptions,
    playback, and StopResponse stay in AgentSession. Menu events are informational
    and never execute actions.

    Example:
        async with AMD(session) as amd:
            result = await amd.execute()

    Args:
        session: Started session whose participant audio and client-side EOT to use.
        llm: Classification and menu model. When omitted, use
            ``google/gemini-3.1-flash-lite`` if LiveKit Cloud credentials are
            available; otherwise use the current Agent's LLM. Pass None to
            always use the current Agent's LLM. A string selects a LiveKit
            Inference model. Supplied models stay open when AMD completes.
        stt: Optional second STT model. When omitted, use ``cartesia/ink-whisper``
            if LiveKit Cloud credentials are available; otherwise use only the
            session transcript. Pass None to always use only the session
            transcript. A string selects a LiveKit Inference model.
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
        machine_silence_threshold: Continuous participant silence before a machine
            prediction releases the turn. Includes silence before and during
            classification. Human and initial uncertain predictions do not wait.
            Set to zero to disable the extra silence wait.
        max_uncertain_turns: Consecutive uncertain predictions before completion.
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
        screening_instructions: str = DEFAULT_SCREENING_INSTRUCTIONS,
        voicemail_instructions: str = DEFAULT_VOICEMAIL_INSTRUCTIONS,
        ivr_instructions: str = DEFAULT_IVR_INSTRUCTIONS,
        idle_timeout: float = 10.0,
        voicemail_idle_timeout: float = 60.0,
        timeout: float = 120.0,
        inference_timeout: float = 1.5,
        machine_silence_threshold: float = MACHINE_SILENCE_THRESHOLD,
        max_uncertain_turns: int = 3,
    ) -> None:
        super().__init__()
        if (
            min(idle_timeout, voicemail_idle_timeout, timeout, inference_timeout) <= 0
            or max_uncertain_turns < 1
        ):
            raise ValueError("amd timeouts and max_uncertain_turns must be positive")
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
            AMDCategory.HUMAN: _HUMAN_INSTRUCTIONS,
        }
        self._fsm = _AMDFSM(
            idle_timeout=idle_timeout,
            voicemail_idle_timeout=voicemail_idle_timeout,
            timeout=timeout,
            inference_timeout=inference_timeout,
            machine_silence_threshold=machine_silence_threshold,
            max_uncertain_turns=max_uncertain_turns,
        )
        self._transcript_grace_period = min(0.5, inference_timeout)
        self._session_id = uuid.uuid4().hex
        self._control_prefix = f"amd_{uuid.uuid4().hex}_"
        self._voicemail_handle: SpeechHandle | None = None
        self._voicemail_audio_start = 0
        self._decision_changed = asyncio.Event()
        self._active: _Run | None = None
        self._transcripts: dict[int, TurnTranscript] = {}
        self._classifier_task: asyncio.Task[None] | None = None
        self._menu_task: asyncio.Task[None] | None = None
        self._finishing: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._timer: asyncio.TimerHandle | None = None
        self._speeches: set[SpeechHandle] = set()

    @property
    def _run(self) -> _Run:
        if self._active is None:
            raise RuntimeError("enter AMD before use")
        return self._active

    @property
    def enabled(self) -> bool:
        return self._fsm.enabled

    @property
    def started(self) -> bool:
        return self._fsm.started

    async def __aenter__(self) -> AMD:
        if self._fsm.entered:
            raise RuntimeError("use a new AMD instance for each run")
        activity = self._session._activity
        if activity is None:
            raise RuntimeError("start AgentSession before entering AMD")
        if isinstance(activity.llm, llm.RealtimeModel):
            raise ValueError("amd does not support realtime models")
        if self._session.amd:
            raise RuntimeError("amd is already active")
        if self._session.options.ivr_detection or self._session._ivr_activity is not None:
            raise ValueError("disable session-level ivr_detection when using AMD")
        model = self._llm if self._llm is not None else activity.llm
        if not isinstance(model, llm.LLM):
            raise ValueError("amd requires an LLM for classification")
        self._active = _Run(
            agent=activity._agent,
            llm=model,
            completion=asyncio.get_running_loop().create_future(),
            transcript=self._new_transcript(),
        )
        self._fsm.enter()
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
            logger.error("amd task failed", exc_info=error)
            self._finish(AMDReason.INTERNAL_ERROR)

    async def execute(self) -> AMDCompletedEvent:
        return await asyncio.shield(self._run.completion)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._fsm.entered:
            return
        self._finish(AMDReason.CANCELLED)
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
            if not self._fsm.finished:
                self._start_listening()
        except RuntimeError:
            self._finish(AMDReason.PARTICIPANT_MISSING)

    def _start_listening(self) -> None:
        if self._fsm.finished or self.started:
            return
        self._fsm.start(time.monotonic())
        self._rearm_idle()
        logger.info("amd listening", extra={"session_id": self._session_id})

    def _on_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._participant_identity:
            self._finish(AMDReason.PARTICIPANT_DISCONNECTED)

    def _new_transcript(self) -> TurnTranscript:
        return TurnTranscript(
            self._stt,
            self._session.conn_options.stt_conn_options,
            self._on_transcript_update,
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.started:
            self._run.transcript.push_audio(frame)

    def _on_transcript_update(self, transcript: TurnTranscript) -> None:
        if transcript.turn_id is not None:
            self._fsm.transcript_updated(transcript.turn_id, transcript.snapshot())

    def notify_dtmf_sent(self, digits: str) -> None:
        """Report digits after their local send succeeds, in send order.

        The built-in send_dtmf_events tool calls this automatically. Custom
        senders must call it after publish_dtmf returns successfully. Report
        each digit separately when a sequence can fail or be cancelled midway.
        The next client-side EOT includes these digits in classification context.
        This does not send DTMF to the participant or change the AMD stage.
        Calls outside an active AMD run are ignored.
        """
        self._fsm.dtmf_sent(digits)

    def _on_user_speech_started(self) -> None:
        if self.started:
            self._fsm.speech_started(time.monotonic())
            if activity := self._session._activity:
                activity._pause_authorization()
            self._rearm_idle()

    def _on_user_speech_ended(self, silence_duration: float) -> None:
        self._state_changed(self._fsm.speech_ended(time.monotonic(), silence_duration))

    def _on_transcript(self, text: str) -> None:
        if self.started:
            self._run.transcript.add_session_text(text)

    def _on_end_of_turn(self, info: _EndOfTurnInfo) -> None:
        if not self.started:
            return
        if activity := self._session._activity:
            activity._pause_authorization()
        run = self._run
        transcript = run.transcript
        transcript.commit(self._fsm.turn_id + 1, info.new_transcript)
        turn_id = self._fsm.commit_turn(
            transcript.snapshot(), time.monotonic(), info.metrics.end_of_turn_delay or 0
        )
        info.amd_turn_id = turn_id
        self._transcripts[turn_id] = transcript
        run.transcript = self._new_transcript()
        if not transcript.ready.is_set() and transcript.pending:
            self._spawn(self._wait_for_transcript(turn_id, transcript))
            self._rearm_idle()
        else:
            self._start_classification(turn_id, transcript)

    async def _wait_for_transcript(self, turn_id: int, transcript: TurnTranscript) -> None:
        try:
            await asyncio.wait_for(transcript.ready.wait(), self._transcript_grace_period)
        except asyncio.TimeoutError:
            pass
        self._start_classification(turn_id, transcript)

    def _start_classification(self, turn_id: int, transcript: TurnTranscript) -> None:
        request, events = self._fsm.transcript_ready(
            turn_id, transcript.snapshot(), time.monotonic()
        )
        if request is not None:
            if self._classifier_task is not None:
                self._classifier_task.cancel()
            if self._menu_task is not None:
                self._menu_task.cancel()
            self._classifier_task = self._spawn(self._classify(turn_id, request))
        self._state_changed(events)

    async def _classify(self, turn_id: int, request: ClassifyRequest) -> None:
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=_inference.CLASSIFY_PROMPT)
        chat_ctx.add_message(role="user", content=request.model_dump_json(exclude_none=True))
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(_inference.classify(self._run.llm, chat_ctx), 30)
            now = time.monotonic()
            events = self._fsm.prediction_received(turn_id, result.category, now, now - started)
        except Exception as exc:
            logger.warning(
                "amd classification failed",
                extra={"turn_id": turn_id, "error_type": type(exc).__name__},
            )
            events = self._fsm.inference_failed(turn_id, time.monotonic())
        else:
            logger.debug(
                "amd classification",
                extra={"turn_id": turn_id, "raw_category": result.category.value},
            )
        self._state_changed(events)

    def _state_changed(self, events: list[_AMDEvent]) -> None:
        completed = False
        try:
            for event in events:
                if isinstance(event, AMDCompletedEvent):
                    completed = True
                    continue
                if event.state_changed and (activity := self._session._activity):
                    activity._cancel_preemptive_generation()
                if (
                    self.started
                    and event.category == AMDCategory.MACHINE_IVR
                    and event.reason in {AMDReason.PREDICTION, AMDReason.LATE_PREDICTION}
                ):
                    self._menu_task = self._spawn(
                        self._extract_menu(event.turn_id, event.transcript)
                    )
                self._emit_prediction(event)
        except Exception:
            logger.exception("amd prediction release failed")
            self._fsm.finish(AMDReason.INTERNAL_ERROR)
            completed = True
        self._decision_changed.set()
        self._rearm_idle()
        if completed and self._finishing is None:
            self._finishing = asyncio.create_task(self._cleanup())

    def _emit_prediction(self, event: AMDPredictionEvent) -> None:
        self.emit("amd_prediction", event)
        if (host := self._session._session_host) is not None:
            host._on_amd_prediction(event)
        logger.info(
            "amd prediction",
            extra={
                "turn_id": event.turn_id,
                "category": event.category.value,
                "reason": event.reason,
                "session_id": self._session_id,
            },
        )

    async def _extract_menu(self, turn_id: int, transcript: str) -> None:
        started = time.monotonic()
        try:
            menu = await asyncio.wait_for(_inference.extract_menu(self._run.llm, transcript), 5)
        except Exception as exc:
            logger.debug("amd menu extraction failed", extra={"error_type": type(exc).__name__})
            return
        if self._fsm.finished or not (menu.menu or menu.options):
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

    async def _wait_for_decision(self, turn_id: int) -> AMDPredictionEvent:
        while (decision := self._fsm.decision(turn_id)) is None:
            self._decision_changed.clear()
            await self._decision_changed.wait()
        return decision

    async def _should_reply(self, info: _EndOfTurnInfo, chat_ctx: llm.ChatContext) -> bool:
        """Wait for the turn's decision, then add stage instructions when a reply is allowed."""
        turn_id = info.amd_turn_id
        if turn_id is not None and self._fsm.has_turn(turn_id):
            await self._wait_for_decision(turn_id)
            if not self._fsm.finished and self._session.current_agent is not self._run.agent:
                self._finish(AMDReason.AGENT_CHANGED)
                return False
        decision = self._fsm.authorize_reply(turn_id)
        if not decision.allow:
            self._rearm_idle()
            return False
        if decision.instructions_for is not None:
            self._add_instructions(chat_ctx, decision.instructions_for, turn_id)
        if self._fsm.has_turn(turn_id) and self.started and (activity := self._session._activity):
            activity._resume_authorization()
        return True

    def _add_instructions(
        self, chat_ctx: llm.ChatContext, stage: AMDCategory, turn_id: int | None
    ) -> None:
        content = self._instructions[stage]
        if stage is AMDCategory.MACHINE_IVR and self._fsm.voicemail_message_played:
            content += "\nThe voicemail message already played locally."
        chat_ctx.add_message(
            id=f"{self._control_prefix}{turn_id}",
            role="user",
            content=content,
            extra={"amd_run": self._session_id, "amd_stage": self._fsm.category.value},
        )

    def _maybe_inject_dtmf_tool(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        if self.started and self._fsm.category == AMDCategory.MACHINE_IVR:
            from ...beta.tools.send_dtmf import send_dtmf_events

            if not any(tool.id == send_dtmf_events.id for tool in tools):
                return [*tools, send_dtmf_events]
        return tools

    def _on_reply_created(self, handle: SpeechHandle, turn_id: int | None) -> None:
        if (
            self.started
            and turn_id == self._fsm.turn_id
            and self._fsm.category == AMDCategory.MACHINE_VM
        ):
            self._voicemail_handle = handle
            output = self._session.output.audio
            self._voicemail_audio_start = output.captured_playout_segments if output else 0

    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        if self._fsm.finished:
            return
        self._speeches.add(event.speech_handle)
        event.speech_handle.add_done_callback(self._on_speech_done)
        self._rearm_idle()

    def _on_speech_done(self, handle: SpeechHandle) -> None:
        self._speeches.discard(handle)
        if (
            handle is self._voicemail_handle
            and not handle.interrupted
            and handle.exception() is None
        ):
            output = self._session.output.audio
            if output and output.captured_playout_segments > self._voicemail_audio_start:
                self._fsm.voicemail_played()
        self._rearm_idle()

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        self._rearm_idle()

    def _on_false_interruption(self, event: AgentFalseInterruptionEvent) -> None:
        # AgentSession clears its pause state after it emits this event.
        asyncio.get_running_loop().call_soon(self._rearm_idle)

    def _on_deadline(self) -> None:
        self._state_changed(self._fsm.tick(time.monotonic()))

    def _rearm_idle(self) -> None:
        activity = self._session._activity
        self._fsm.update_idle(
            time.monotonic(),
            session_busy=bool(self._speeches) or activity is None or activity._is_busy,
        )
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if (at := self._fsm.next_deadline) is not None:
            self._timer = asyncio.get_running_loop().call_later(
                max(0, at - time.monotonic()), self._on_deadline
            )

    def _finish(self, reason: AMDReason) -> None:
        self._state_changed([self._fsm.finish(reason)])

    async def _cleanup(self) -> None:
        try:
            await aio.cancel_and_wait(*self._tasks)
            transcripts = [*self._transcripts.values(), self._run.transcript]
            close_tasks = [transcript.aclose() for transcript in transcripts]
            if self._owns_stt and self._stt is not None:
                close_tasks.append(self._stt.aclose())
            if self._owns_llm and self._llm is not None:
                close_tasks.append(self._llm.aclose())
            for error in await asyncio.gather(*close_tasks, return_exceptions=True):
                if isinstance(error, BaseException):
                    logger.warning(
                        "amd resource cleanup failed", extra={"error_type": type(error).__name__}
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
                if self._fsm.category == AMDCategory.MACHINE_UNAVAILABLE:
                    activity._cancel_pending_replies()
                activity._resume_authorization()
            if self._session._amd is self:
                self._session._amd = None
        finally:
            result = self._fsm.completion()
            self._run.completion.set_result(result)
            self.emit("amd_completed", result)
