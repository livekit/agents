from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING

from opentelemetry import trace

from livekit import rtc

from ...inference import LLM as _InferenceLLM, LLMModels
from ...job import get_job_context
from ...llm import LLM as _LLM
from ...log import logger
from ...telemetry import trace_types, tracer
from ...utils import aio
from .classifier import (
    AMD_PROMPT,
    HUMAN_SILENCE_THRESHOLD,
    HUMAN_SPEECH_THRESHOLD,
    MACHINE_SILENCE_THRESHOLD,
    NO_SPEECH_THRESHOLD,
    TIMEOUT,
    AMDCategory,
    AMDResult,
    _AMDClassifier,
)

if TYPE_CHECKING:
    from ...llm import LLM
    from ...stt import STT
    from ..agent_session import AgentSession

VERIFIED_LLM_MODELS = {
    "google/gemini-3.1-flash-lite-preview",
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

VERIFIED_STT_MODELS = {
    "deepgram/nova-3",
    "assemblyai/universal-streaming-multilingual",
    "cartesia/ink-whisper",
}


class AMD:
    """Answering Machine Detection (AMD).

    Detects whether an outbound call is answered by a human or a machine.

    Listens to the call greeting and uses an LLM to classify it into one of
    the following categories:

    - ``human``: a real person answered.
    - ``machine-ivr``: an IVR / DTMF menu prompt was detected.
    - ``machine-vm``: a voicemail greeting where leaving a message is possible.
    - ``machine-unavailable``: the mailbox is full or not set up; leaving a message is not possible.
    - ``uncertain``: the transcript is ambiguous and could not be classified.

    AMD should be started before the SIP participant is created so no audio is
    missed. Timers begin when the participant's audio track is subscribed.

    The recommended pattern is the async context manager::

        async with AMD(session, llm="openai/gpt-4.1-mini") as detector:
            await ctx.api.sip.create_sip_participant(...)
            await ctx.wait_for_participant(identity=participant_identity)
            result = await detector.execute()

    Args:
        session: The :class:`AgentSession` to wire AMD to. Can also be passed
            later via :meth:`start`.
        llm: LLM used for greeting classification. Accepts an :class:`LLM`
            instance, an inference model string (e.g. ``"openai/gpt-4.1-mini"``),
            or ``None`` to fall back to the session's own LLM.
        interrupt_on_machine: If ``True`` (default), interrupt any pending
            agent speech immediately when a machine is detected.
        ivr_detection: If ``True`` (default), automatically start IVR
            navigation when a ``machine-ivr`` result is returned.
        participant_identity: If set, only this participant's audio track
            subscription triggers the detection timers. If ``None``, the first
            remote audio track wins.
        stt: Optional STT to use for transcript generation. Required when the
            session uses no STT (e.g. a realtime model). If ``None``
            and the session has its own STT, AMD reuses those transcripts.
        max_hold_duration: Maximum seconds the IVR navigator will stay in hold
            mode before auto-exiting. Passed through to :class:`IVRActivity`.
        suppress_compatibility_warning: If ``True``, suppress model compatibility warning messages.
    """

    def __init__(
        self,
        session: AgentSession | None = None,
        *,
        llm: LLM | LLMModels | str | None = "google/gemini-3.1-flash-lite-preview",
        stt: STT | None = "cartesia/ink-whisper",
        interrupt_on_machine: bool = True,
        ivr_detection: bool = True,
        human_speech_threshold: float = HUMAN_SPEECH_THRESHOLD,
        human_silence_threshold: float = HUMAN_SILENCE_THRESHOLD,
        machine_silence_threshold: float = MACHINE_SILENCE_THRESHOLD,
        no_speech_threshold: float = NO_SPEECH_THRESHOLD,
        timeout: float = TIMEOUT,
        prompt: str = AMD_PROMPT,
        participant_identity: str | None = None,
        max_hold_duration: float = 300.0,
        suppress_compatibility_warning: bool = False,
    ) -> None:
        self._llm_config = llm
        self._session: AgentSession | None = session
        self._classifier: _AMDClassifier | None = None
        self._result: AMDResult | None = None
        self._closed = False
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection
        self._span: trace.Span | None = None
        self._suppress_compatibility_warning = suppress_compatibility_warning

        self._human_speech_threshold = human_speech_threshold
        self._human_silence_threshold = human_silence_threshold
        self._machine_silence_threshold = machine_silence_threshold
        self._no_speech_threshold = no_speech_threshold
        self._timeout = timeout
        self._prompt = prompt

        self._participant_identity = participant_identity
        self._stt = stt
        self._max_hold_duration = max_hold_duration

        stt_model = self._stt.model.lower()
        if (
            self._stt is not None
            and all(
                stt_model != candidate.lower() and stt_model not in candidate.lower()
                for candidate in VERIFIED_STT_MODELS
            )
            and not self._suppress_compatibility_warning
        ):
            logger.warning(
                "stt model %s might not be compatible with amd, proceed with caution",
                self._stt.model,
            )

        self._stt_task: asyncio.Task[None] | None = None
        self._audio_ch: aio.Chan[rtc.AudioFrame] | None = None

        self._track_subscribed_handler: bool = False

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    @property
    def started(self) -> bool:
        return self._classifier is not None and self._classifier.started

    async def start(self, session: AgentSession) -> None:
        """Wire AMD to the given session."""
        self._session = session
        await self._run(session)

    async def execute(self) -> AMDResult:
        """Run AMD and return the result.

        While executing, speech playout authorization is locked. Once the
        result is available, authorization is resumed and automatic actions
        (interrupt on machine, ivr detection) are applied based on the
        configured options.
        """
        if self._session is None:
            raise RuntimeError("AMD is not wired to a session, call start() first")

        if self._classifier is not None:
            await self._classifier._verdict_ready.wait()
        if self._result is None:
            raise RuntimeError("AMD was closed before a result was available")

        result = self._result

        if result.is_machine and self._interrupt_on_machine:
            await self._session.interrupt(force=True)

        if result.category == AMDCategory.MACHINE_IVR and self._ivr_detection:
            await self._session._start_ivr_detection(
                transcript=result.transcript,
                max_hold_duration=self._max_hold_duration,
            )

        # eagerly resume so agent can speak immediately to a human
        if self._session._activity is not None:
            self._session._activity._resume_authorization()

        return result

    async def __aenter__(self) -> AMD:
        if self._session is not None:
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
        if (
            self._audio_ch is not None
            and not self._audio_ch.closed
            and self._classifier is not None
        ):
            self._audio_ch.send_nowait(frame)

    def _on_user_speech_started(self) -> None:
        if self._classifier is not None:
            self._classifier.on_user_speech_started()

    def _on_user_speech_ended(self, silence_duration: float) -> None:
        if self._classifier is not None:
            self._classifier.on_user_speech_ended(silence_duration)

    def _on_transcript(self, text: str) -> None:
        if self._classifier is not None:
            self._classifier.push_text(text)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._unsubscribe_track_event()

        if self._stt_task is not None:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                pass
            self._stt_task = None

        if self._classifier is not None:
            self._classifier.off("amd_result", self._on_amd_result)
            await self._classifier.close()
            self._classifier = None

        self._end_span()

        if self._session is not None and self._session._activity is not None:
            self._session._activity._resume_authorization()

        if self._session is not None:
            self._session._amd = None

    # endregion

    # region: internal methods

    async def _run(self, session: AgentSession) -> None:
        if self._classifier is not None:
            logger.warning("AMD already running, skipping")
            return

        self._session = session
        self._classifier = self._resolve_classifier(self._llm_config, session)
        if self._classifier is None:
            raise ValueError(
                "AMD classifier could not be resolved, please provide a compatible model"
            )
        self._classifier.on("amd_result", self._on_amd_result)
        self._closed = False
        self._result = None

        if session.options.ivr_detection:
            logger.warning("session level ivr_detection will be disabled when AMD is used")
            session.options.ivr_detection = False

        if session._ivr_activity is not None:
            logger.warning(
                "session-level IVR detection was already started, "
                "closing it so AMD can manage the IVR lifecycle"
            )
            await session._ivr_activity.aclose()
            session._ivr_activity = None

        session._amd = self

        # start the classifier first and the timers later when the track is subscribed
        self._classifier.start()
        self._start_span()
        if session._activity is not None:
            session._activity._pause_authorization()

        self._stt_task = asyncio.create_task(self._setup(session), name="amd_setup")

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if self._participant_identity is not None:
            if participant.identity != self._participant_identity:
                return

        self._unsubscribe_track_event()

        if self._classifier is not None:
            self._classifier.start_timers()

    def _unsubscribe_track_event(self) -> None:
        """Unsubscribe from the track subscribed event."""
        if (
            self._track_subscribed_handler
            and self._session is not None
            and self._session._room_io is not None
        ):
            self._session._room_io.room.off("track_subscribed", self._on_track_subscribed)
            self._track_subscribed_handler = False

    async def _setup(self, session: AgentSession) -> None:
        if not self._closed:
            if session._room_io is not None:
                room = session._room_io.room
                room.on("track_subscribed", self._on_track_subscribed)
                self._track_subscribed_handler = True

                for participant in room.remote_participants.values():
                    for pub in participant.track_publications.values():
                        if (
                            pub.subscribed
                            and pub.track is not None
                            and pub.track.kind == rtc.TrackKind.KIND_AUDIO
                        ):
                            if (
                                self._participant_identity is None
                                or participant.identity == self._participant_identity
                            ):
                                self._on_track_subscribed(
                                    pub.track,
                                    pub,
                                    participant,
                                )
                                break
            else:
                logger.warning(
                    "session room_io unavailable, starting amd timers immediately as fallback"
                )
                if self._classifier is not None:
                    self._classifier.start_timers()

        if self._stt is not None and not self._closed:
            logger.debug("starting amd stt pipeline")
            await self._run_stt()

    async def _run_stt(self) -> None:
        assert self._stt is not None
        assert self._classifier is not None

        self._audio_ch = aio.Chan[rtc.AudioFrame]()
        stt_stream = self._stt.stream()
        try:

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
                        and self._classifier is not None
                        and (text := event.alternatives[0].text)
                    ):
                        self._classifier.push_text(text, source="amd_stt")

            tasks = [
                asyncio.create_task(_send(self._audio_ch), name="amd_stt_send"),
                asyncio.create_task(_receive(), name="amd_stt_receive"),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await aio.cancel_and_wait(*tasks)
        finally:
            await stt_stream.aclose()

    def _on_amd_result(self, result: AMDResult) -> None:
        self._result = result
        if self._classifier is not None:
            self._classifier.end_input()
        if self._audio_ch is not None:
            self._audio_ch.close()

        if self._span is not None:
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

    def _start_span(self) -> None:
        if self._span is not None:
            return
        context = (
            self._session._root_span_context
            if self._session is not None and self._session._root_span_context is not None
            else None
        )
        self._span = tracer.start_span("amd", context=context)

    def _end_span(self) -> None:
        if self._span is None:
            return
        self._span.end()
        self._span = None

    def _resolve_classifier(
        self,
        llm: LLM | LLMModels | str | None,
        session: AgentSession,
    ) -> _AMDClassifier | None:
        _llm: _InferenceLLM | _LLM | None = None
        if isinstance(llm, str):
            _llm = _InferenceLLM(llm)
        elif isinstance(llm, _LLM):
            _llm = llm
        elif (candidate := session.llm) is not None and isinstance(candidate, _LLM):
            _llm = candidate

        model_name = _llm.model.lower()
        if (
            all(
                model_name != candidate.lower() and model_name not in candidate.lower()
                for candidate in VERIFIED_LLM_MODELS
            )
            and not self._suppress_compatibility_warning
        ):
            logger.warning(
                "llm model %s might not be compatible with amd, proceed with caution", model_name
            )

        if _llm is not None:
            return _AMDClassifier(
                _llm,
                human_speech_threshold=self._human_speech_threshold,
                human_silence_threshold=self._human_silence_threshold,
                machine_silence_threshold=self._machine_silence_threshold,
                no_speech_threshold=self._no_speech_threshold,
                timeout=self._timeout,
                prompt=self._prompt,
                source="stt" if self._stt is None else "amd_stt",
            )

        return None

    # endregion
