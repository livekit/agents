from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from opentelemetry import trace

from livekit import rtc

from ..._exceptions import APIError
from ...inference import LLM as _InferenceLLM, STT as _InferenceSTT, LLMModels
from ...job import get_job_context
from ...llm import LLM as _LLM, RealtimeModel as _RealtimeModel
from ...log import logger
from ...stt import STT as _STT
from ...telemetry import trace_types, tracer
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.participant import (
    wait_for_participant_attribute,
    wait_for_track_publication,
)
from ..speech_handle import SpeechHandle
from .classifier import (
    AMD_PROMPT,
    HUMAN_SILENCE_THRESHOLD,
    HUMAN_SPEECH_THRESHOLD,
    MACHINE_SILENCE_THRESHOLD,
    MAX_ENDPOINTING_DELAY,
    NO_SPEECH_THRESHOLD,
    TIMEOUT,
    AMDCategory,
    AMDPredictionEvent,
    _AMDClassifier,
)

if TYPE_CHECKING:
    from ...llm import LLM
    from ...stt import STT
    from ..agent_session import AgentSession
    from ..audio_recognition import _EndOfTurnInfo

EVALUATED_LLM_MODELS: set[str] = {
    "google/gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview",
    "openai/gpt-4.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.1",
    "openai/gpt-4o",
    "openai/gpt-5.1-chat-latest",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-5.2-chat-latest",
    "google/gemini-2.5-flash-lite",
}

EVALUATED_STT_MODELS: set[str] = {
    "deepgram/nova-3",
    "assemblyai/universal-streaming-multilingual",
    "cartesia/ink-whisper",
}

_SIP_CALL_STATUS_ATTR = "sip.callStatus"
_SIP_CALL_STATUS_ACTIVE = "active"

# A predefined AMD response: a literal string spoken via ``session.say``, or a callable
# invoked with the triggering prediction that returns a string (spoken), a SpeechHandle
# (the caller drove its own say/generate_reply), or None to skip playout. Mirrors the
# ``with_filler`` source shape.
_AMDMessage = str | Callable[[AMDPredictionEvent], SpeechHandle | str | None]

_MessagePlayback = Literal["played", "interrupted", "not_played"]


class DetectionOptions(TypedDict, total=False):
    human_speech_threshold: float
    human_silence_threshold: float
    machine_silence_threshold: float
    no_speech_threshold: float
    timeout: float
    prompt: str


_DEFAULT_DETECTION_OPTIONS: DetectionOptions = {
    "human_speech_threshold": HUMAN_SPEECH_THRESHOLD,
    "human_silence_threshold": HUMAN_SILENCE_THRESHOLD,
    "machine_silence_threshold": MACHINE_SILENCE_THRESHOLD,
    "no_speech_threshold": NO_SPEECH_THRESHOLD,
    "timeout": TIMEOUT,
    "prompt": AMD_PROMPT,
}


class AMD(EventEmitter[Literal["amd_prediction"]]):
    """Answering Machine Detection (AMD).

    Detects whether an outbound call is answered by a human or a machine.

    Listens to the call greeting and uses an LLM to classify it into one of
    the following categories:

    - ``human``: a real person answered.
    - ``machine-ivr``: an IVR / DTMF menu prompt was detected.
    - ``machine-screening``: a call-screening prompt asking the caller to identify
      themselves before being connected.
    - ``machine-vm``: a voicemail greeting where leaving a message is possible.
    - ``machine-unavailable``: the mailbox is full or not set up; leaving a message is not possible.
    - ``uncertain``: the transcript is ambiguous and could not be classified.

    AMD should be started before the SIP participant is created so no audio
    is missed. The overall detection-timeout budget starts when the
    participant's audio track is subscribed (so AMD cannot hang if the call
    never connects).

    For SIP participants, the no-speech timer and
    audio/transcript processing are deferred until ``sip.callStatus ==
    "active"`` so pre-answer audio (ringback, carrier early media, dialtone)
    does not poison the classifier or burn the no-speech budget.

    Call screening is handled internally: when a ``machine-screening`` prompt is detected
    AMD auto-plays ``screening_message`` and keeps listening for the next greeting; when a
    voicemail is detected it auto-plays ``voicemail_message``. If no ``screening_message``
    was provided, ``machine-screening`` is returned as the terminal verdict (AMD can't
    advance the prompt). Otherwise only the terminal verdict (``human`` / ``machine-vm`` /
    ``machine-unavailable`` / ``uncertain``) is surfaced — ``execute()`` returns it once,
    carrying ``screening_detected`` and ``message_playback``.

    The recommended pattern is the async context manager::

        async with AMD(session, stt="cartesia/ink-whisper") as detector:
            await ctx.api.sip.create_sip_participant(...)
            await ctx.wait_for_participant(identity=participant_identity)
            result = await detector.execute()

    Args:
        session: The :class:`AgentSession` to wire AMD to.
        llm: LLM used for greeting classification. Accepts an :class:`LLM`
            instance or an inference model string (e.g. ``"openai/gpt-4.1-mini"``).
            When omitted, AMD reuses the session's own LLM (``session.llm``).
        interrupt_on_machine: If ``True`` (default), interrupt any pending
            agent speech immediately when a machine is detected.
        ivr_detection: If ``True`` (default), automatically start IVR
            navigation when a ``machine-ivr`` result is returned.
        participant_identity: If set, AMD listens only to this participant's
            audio track. If omitted, the first remote audio track wins and
            the publisher is resolved from the track sid.
        stt: STT used for transcript generation. Accepts an :class:`STT`
            instance or an inference model string (e.g. ``"cartesia/ink-whisper"``).
            When omitted, AMD reuses the session's existing STT transcripts (no
            dedicated stream). Pass ``"cartesia/ink-whisper"`` for the dedicated,
            AMD-tuned model.
        screening_message: Response played when a ``machine-screening`` prompt is
            detected. A literal string (spoken via ``session.say``) or a callable
            ``(prediction) -> SpeechHandle | str | None`` (return ``None`` to skip).
            When omitted, a ``machine-screening`` prompt is returned as the terminal
            verdict instead of being responded to.
        voicemail_message: Message played when a voicemail is detected. Same shape as
            ``screening_message``.
        suppress_compatibility_warning: If ``True``, do not log a warning when
            the resolved STT or LLM is not among the bundled AMD-tested model
            strings. Has no effect on classification behavior.
        detection_options: Optional overrides for timing thresholds and the AMD
            classification prompt (see :class:`DetectionOptions`). When
            omitted, library defaults apply.
        wait_until_finished: If ``True``, once any speech has been heard the
            ``detection_timeout`` no longer forces emission — AMD will keep
            waiting for the post-speech silence and a positive end-of-turn
            from the session's turn detector before emitting. Useful for
            outbound voicemail flows where leaving a message early would
            overlap the greeting. ``no_speech_timeout`` (uncertain) still fires
            normally (no audio at all means there is nothing to wait for).
            Defaults to ``False``.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        llm: NotGivenOr[LLM | LLMModels | str] = NOT_GIVEN,
        stt: NotGivenOr[STT | str] = NOT_GIVEN,
        interrupt_on_machine: bool = True,
        ivr_detection: bool = True,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        screening_message: NotGivenOr[_AMDMessage] = NOT_GIVEN,
        voicemail_message: NotGivenOr[_AMDMessage] = NOT_GIVEN,
        suppress_compatibility_warning: bool = False,
        detection_options: NotGivenOr[DetectionOptions] = NOT_GIVEN,
        wait_until_finished: bool = False,
    ) -> None:
        super().__init__()

        self._llm_config: NotGivenOr[LLM | LLMModels | str] = llm
        self._session: AgentSession = session
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection
        self._wait_until_finished = wait_until_finished
        self._suppress_compatibility_warning = suppress_compatibility_warning
        self._participant_identity: NotGivenOr[str] = participant_identity
        self._screening_message = screening_message
        self._voicemail_message = voicemail_message
        self._stt: NotGivenOr[_STT] = _InferenceSTT(stt) if isinstance(stt, str) else stt

        self._classifier: _AMDClassifier | None = None
        self._result: AMDPredictionEvent | None = None
        self._screening_detected = False
        self._last_playback: _MessagePlayback = "not_played"
        self._terminal_ready = asyncio.Event()
        self._closed = False
        self._span: trace.Span | None = None

        self._opts: DetectionOptions = (
            {**_DEFAULT_DETECTION_OPTIONS, **detection_options}
            if is_given(detection_options)
            else _DEFAULT_DETECTION_OPTIONS
        )

        if not self._suppress_compatibility_warning and is_given(self._stt):
            _warn_if_not_evaluated(self._stt.model, EVALUATED_STT_MODELS, model_kind="stt")

        self._setup_task: asyncio.Task[None] | None = None
        self._sip_answer_task: asyncio.Task[None] | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._audio_ch: aio.Chan[rtc.AudioFrame] | None = None

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    @property
    def started(self) -> bool:
        return self._classifier is not None and self._classifier.listening

    async def execute(self) -> AMDPredictionEvent:
        """Run AMD and return the terminal result.

        While executing, speech playout authorization is locked (except while AMD plays a
        screening / voicemail message). Screening and IVR turns are driven internally; this
        returns once a terminal verdict (``human`` / ``machine-vm`` / ``machine-unavailable``
        / ``uncertain``) is reached, then resumes authorization so the agent can speak.
        """
        await self._terminal_ready.wait()

        if not self._result:
            raise RuntimeError("amd closed before a result was available")

        # eagerly resume so the agent can speak immediately to a human
        if self._session._activity:
            self._session._activity._resume_authorization()

        return self._result

    def _on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        """Forward EOT to the classifier and signal whether AMD is taking over this turn.

        Returns ``True`` only when AMD is handling the turn itself — a machine verdict
        (screening / ivr / voicemail / unavailable) under ``interrupt_on_machine`` — so the
        agent's auto-reply does not race AMD's interrupt or message playout. Human and
        uncertain turns are left to the session's normal behavior; whether the agent replies
        (or the caller drives a reply after ``execute()``) is the caller's decision, not ours.

        Reads the *current* per-turn verdict (``_verdict_result``), not the terminal
        ``self._result`` — during an internal screening turn the terminal result isn't set
        yet, but the screening verdict is, which is exactly the turn we must consume.
        """
        if self._closed or not self._classifier:
            return False
        self._classifier.on_end_of_turn()
        verdict = self._classifier._verdict_result
        if not (self._interrupt_on_machine and verdict is not None and verdict.is_machine):
            return False
        logger.debug(
            "skipping auto reply: AMD is handling a machine turn",
            extra={"category": verdict.category.value, "transcript": info.new_transcript},
        )
        return True

    async def __aenter__(self) -> AMD:
        await self._run(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # region: lifecycle hooks (called by AudioRecognition)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not (self._classifier and self._classifier.listening):
            return
        if self._audio_ch and not self._audio_ch.closed:
            self._audio_ch.send_nowait(frame)

    def _on_user_speech_started(self) -> None:
        if self._classifier:
            self._classifier.on_user_speech_started()

    def _on_user_speech_ended(self, silence_duration: float) -> None:
        if self._classifier:
            self._classifier.on_user_speech_ended(silence_duration)

    def _on_transcript(self, text: str) -> None:
        if self._classifier:
            self._classifier.push_text(text)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        # unblock execute() if we're torn down before reaching a terminal verdict
        self._terminal_ready.set()

        pending = [
            t for t in (self._sip_answer_task, self._setup_task, self._loop_task) if t is not None
        ]
        if pending:
            await aio.cancel_and_wait(*pending)
        self._sip_answer_task = None
        self._setup_task = None
        self._loop_task = None

        if self._audio_ch and not self._audio_ch.closed:
            self._audio_ch.close()

        if self._classifier:
            await self._classifier.close()
            self._classifier = None

        self._end_span()

        if self._session._activity:
            self._session._activity._resume_authorization()

        self._session._amd = None

    # endregion

    # region: internal methods

    async def _run(self, session: AgentSession) -> None:
        if self._classifier:
            logger.warning("AMD already running, skipping")
            return

        self._session = session
        self._classifier = self._resolve_classifier(session)
        if not self._classifier:
            raise ValueError(
                "AMD classifier could not be resolved, please provide a compatible model"
            )
        self._closed = False
        self._result = None
        self._screening_detected = False
        self._last_playback = "not_played"
        self._terminal_ready.clear()

        if session.options.ivr_detection:
            logger.warning("session level ivr_detection will be disabled when AMD is used")
            session.options.ivr_detection = False

        if session._ivr_activity:
            logger.warning(
                "session-level IVR detection was already started, "
                "closing it so AMD can manage the IVR lifecycle"
            )
            await session._ivr_activity.aclose()
            session._ivr_activity = None

        session._amd = self

        # classifier is dormant until start_detection_timer / start_listening;
        # the listening gate stays closed so pre-setup audio is dropped.
        self._start_span()
        if session._activity:
            session._activity._pause_authorization()

        self._setup_task = asyncio.create_task(self._setup(session), name="amd_setup")
        self._loop_task = asyncio.create_task(self._detection_loop(), name="amd_detection_loop")

    async def _setup(self, session: AgentSession) -> None:
        if self._closed:
            return
        if not session._room_io:
            logger.warning(
                "session room_io unavailable, starting amd timers immediately as fallback"
            )
            if self._classifier:
                self._classifier.start_detection_timer()
                self._classifier.start_listening()
        else:
            room = session._room_io.room
            publication = await wait_for_track_publication(
                room=room,
                identity=self._participant_identity or None,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
            )
            if self._closed or not self._classifier:
                return
            # outer budget runs from track-up so AMD bails out even if the
            # call never reaches the active state
            self._classifier.start_detection_timer()

            if self._participant_identity:
                publisher = room.remote_participants.get(self._participant_identity)
            else:
                publisher = next(
                    (
                        p
                        for p in room.remote_participants.values()
                        if publication.sid in p.track_publications
                    ),
                    None,
                )
            if publisher is None:
                # publisher gone start listening so the no-speech timer settles faster
                self._start_listening()
                return

            if publisher.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                self._sip_answer_task = asyncio.create_task(
                    self._wait_for_sip_answer(room, publisher.identity),
                    name="amd_sip_answer",
                )
            else:
                self._start_listening()

        if is_given(self._stt) and not self._closed:
            logger.debug("starting amd stt pipeline")
            await self._run_stt()

    def _start_listening(self) -> None:
        if self._closed or not self._classifier:
            return
        self._classifier.start_listening()
        logger.debug("call has been answered, AMD starts listening")

    async def _wait_for_sip_answer(self, room: rtc.Room, identity: str) -> None:
        try:
            await wait_for_participant_attribute(
                room,
                identity=identity,
                attribute=_SIP_CALL_STATUS_ATTR,
                value=_SIP_CALL_STATUS_ACTIVE,
            )
        except RuntimeError as e:
            # SIP participant disconnected before going active, default to detection timeout
            logger.debug(
                "AMD: SIP answer wait failed; starting to listen", extra={"reason": str(e)}
            )

        if not self._closed:
            self._start_listening()

    async def _run_stt(self) -> None:
        assert is_given(self._stt)
        assert self._classifier

        self._audio_ch = aio.Chan[rtc.AudioFrame]()

        async with self._stt.stream() as stt_stream:

            async def _send(chan: aio.Chan[rtc.AudioFrame]) -> None:
                async for frame in chan:
                    stt_stream.push_frame(frame)

                stt_stream.end_input()

            async def _receive() -> None:
                from ...stt import SpeechEventType

                async for event in stt_stream:
                    if (
                        event.type == SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                        and self._classifier
                        and (text := event.alternatives[0].text)
                    ):
                        self._classifier.push_text(text, source="amd_stt")

            tasks = [
                asyncio.create_task(_send(self._audio_ch), name="amd_stt_send"),
                asyncio.create_task(_receive(), name="amd_stt_receive"),
            ]
            try:
                await asyncio.gather(*tasks)
            except APIError as e:
                # dedicated STT died silently (the push_text race already covers the case
                # where the session STT simply beats it). Fall back one-way to the session's
                # transcripts so detection can still proceed.
                logger.warning(
                    "amd: dedicated STT failed, falling back to session transcripts",
                    extra={"error": str(e)},
                )
                if self._classifier:
                    self._classifier.switch_source("stt")
            finally:
                await aio.cancel_and_wait(*tasks)

    async def _detection_loop(self) -> None:
        """Drive screening turns internally and surface a single terminal verdict.

        ``machine-screening`` plays ``screening_message`` and keeps listening (but is
        terminal when no ``screening_message`` was supplied — there's no way to advance the
        prompt); ``machine-ivr`` starts IVR navigation and keeps listening. Every other
        category (``human`` / ``machine-vm`` / ``machine-unavailable`` / ``uncertain``) is
        terminal — a voicemail additionally plays ``voicemail_message`` before finishing.
        Only the continue paths need an active prompt/menu to recur, so the loop terminates
        without a turn cap.
        """
        classifier = self._classifier
        if classifier is None:
            return
        try:
            while not self._closed:
                await classifier._verdict_ready.wait()
                if self._closed:
                    return
                verdict = classifier._verdict_result
                if verdict is None:
                    return

                if verdict.category == AMDCategory.MACHINE_SCREENING:
                    self._screening_detected = True
                    if self._interrupt_on_machine:
                        await self._session.interrupt(force=True)
                    if not is_given(self._screening_message):
                        # no way to respond to the prompt, so the screening system won't
                        # advance — surface the screening verdict instead of looping into a
                        # no-speech timeout.
                        self._finish(verdict, "not_played")
                        return
                    # remember the screening playback so a later human verdict reports it
                    self._last_playback = await self._play(self._screening_message, verdict)
                    await classifier.reset()
                    continue

                if verdict.category == AMDCategory.MACHINE_IVR:
                    if self._interrupt_on_machine:
                        await self._session.interrupt(force=True)
                    if self._ivr_detection:
                        await self._session._start_ivr_detection(transcript=verdict.transcript)
                    await classifier.reset()
                    continue

                # terminal verdict
                if verdict.is_machine and self._interrupt_on_machine:
                    await self._session.interrupt(force=True)
                if verdict.category == AMDCategory.MACHINE_VM:
                    # voicemail message is the terminal-relevant one
                    playback = await self._play(self._voicemail_message, verdict)
                else:
                    # human/unavailable/uncertain: reflect the screening message if any
                    playback = self._last_playback
                self._finish(verdict, playback)
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("amd detection loop failed")
            # always release execute(): a hang is worse than surfacing an error verdict.
            if not self._terminal_ready.is_set():
                verdict = classifier._verdict_result or AMDPredictionEvent(
                    speech_duration=0.0,
                    category=AMDCategory.UNCERTAIN,
                    reason="detection_error",
                    transcript="",
                    delay=0.0,
                )
                self._finish(verdict, self._last_playback)

    def _resolve_message_handle(
        self, message: NotGivenOr[_AMDMessage], prediction: AMDPredictionEvent
    ) -> SpeechHandle | None:
        if not is_given(message):
            return None
        msg = cast(_AMDMessage, message)
        source = msg(prediction) if callable(msg) else msg
        if source is None:
            return None
        if isinstance(source, SpeechHandle):
            return source
        return self._session.say(source)

    async def _play(
        self, message: NotGivenOr[_AMDMessage], prediction: AMDPredictionEvent
    ) -> _MessagePlayback:
        """Play a predefined message, taking the floor (auth is paused during detection).

        Returns ``interrupted`` if the callee barged in before playout finished.
        """
        if not is_given(message):
            return "not_played"

        activity = self._session._activity
        if activity is not None:
            activity._resume_authorization()
        try:
            handle = self._resolve_message_handle(message, prediction)
            if handle is None:
                return "not_played"
            await handle.wait_for_playout()
            return "interrupted" if handle.interrupted else "played"
        finally:
            # keep the agent quiet while AMD keeps classifying the next turn
            if activity is not None and not self._closed:
                activity._pause_authorization()

    def _finish(self, result: AMDPredictionEvent, playback: _MessagePlayback) -> None:
        result.screening_detected = self._screening_detected
        result.message_playback = playback
        self._result = result
        logger.info(
            "amd prediction",
            extra={
                "category": result.category.value,
                "reason": result.reason,
                "speech_duration": result.speech_duration,
                "delay": result.delay,
                "transcript": result.transcript,
                "screening_detected": result.screening_detected,
                "message_playback": result.message_playback,
            },
        )
        if self._classifier:
            self._classifier.end_input()
        if self._audio_ch:
            self._audio_ch.close()

        if self._span:
            self._span.set_attributes(
                {
                    trace_types.ATTR_AMD_CATEGORY: result.category.value,
                    trace_types.ATTR_AMD_REASON: result.reason,
                    trace_types.ATTR_AMD_SPEECH_DURATION: result.speech_duration,
                    trace_types.ATTR_AMD_DELAY: result.delay,
                    trace_types.ATTR_AMD_TRANSCRIPT: result.transcript,
                }
            )

        self._end_span()

        try:
            ctx = get_job_context()
            ctx.tagger.add(
                f"lk.amd:{result.category.value}",
                metadata={
                    "category": result.category.value,
                    "speech_duration": result.speech_duration,
                    "reason": result.reason,
                    "transcript": result.transcript,
                    "delay": result.delay,
                },
            )
        except RuntimeError:
            pass

        if (host := self._session._session_host) is not None:
            host._on_amd_prediction(result)

        self.emit("amd_prediction", result)
        self._terminal_ready.set()

    def _start_span(self) -> None:
        if self._span:
            return
        self._span = tracer.start_span("amd", context=self._session._root_span_context)

    def _end_span(self) -> None:
        if not self._span:
            return
        self._span.end()
        self._span = None

    def _resolve_classifier(
        self,
        session: AgentSession,
    ) -> _AMDClassifier | None:
        _llm: _InferenceLLM | _LLM | None = None
        if isinstance(self._llm_config, str):
            _llm = _InferenceLLM(self._llm_config)
        elif isinstance(self._llm_config, _LLM):
            _llm = self._llm_config
        elif (candidate := session.llm) and isinstance(candidate, _LLM):
            _llm = candidate

        if (
            _llm is None
            and not is_given(self._llm_config)
            and isinstance(session.llm, _RealtimeModel)
        ):
            raise ValueError(
                "AMD needs a chat LLM, but the session uses a realtime model. "
                "Pass an explicit `llm` to AMD()."
            )

        if not self._suppress_compatibility_warning:
            _warn_if_not_evaluated(
                _llm.model if _llm else None,
                EVALUATED_LLM_MODELS,
                model_kind="llm",
            )
            # when reusing the session STT (no dedicated stream), warn on the active model
            if not is_given(self._stt):
                active_stt = (
                    session._activity.stt
                    if session._activity and session._activity.stt
                    else session.stt
                )
                if active_stt is not None:
                    _warn_if_not_evaluated(active_stt.model, EVALUATED_STT_MODELS, model_kind="stt")
                else:
                    logger.warning(
                        "amd: no session STT to reuse; pass `stt` or configure a session STT"
                    )

        if _llm:
            max_endpointing_delay = (
                session._activity.max_endpointing_delay
                if session._activity
                else MAX_ENDPOINTING_DELAY
            )
            return _AMDClassifier(
                _llm,
                human_speech_threshold=self._opts["human_speech_threshold"],
                human_silence_threshold=self._opts["human_silence_threshold"],
                machine_silence_threshold=self._opts["machine_silence_threshold"],
                no_speech_threshold=self._opts["no_speech_threshold"],
                timeout=self._opts["timeout"],
                prompt=self._opts["prompt"],
                source="amd_stt" if is_given(self._stt) else "stt",
                wait_until_finished=self._wait_until_finished,
                max_endpointing_delay=max_endpointing_delay,
            )

        return None

    # endregion


def _warn_if_not_evaluated(
    model: str | None,
    evaluated_models: set[str],
    *,
    model_kind: str,
) -> None:
    if not model:
        return

    model = model.lower()
    if all(
        model != candidate.lower() and model not in candidate.lower()
        for candidate in evaluated_models
    ):
        logger.warning(
            "%s model %s hasn't been evaluated with our benchmark, it might not be compatible "
            "with amd. Set `suppress_compatibility_warning=True` to silence this warning.",
            model_kind,
            model,
        )
