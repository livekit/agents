from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Literal, TypedDict

from opentelemetry import trace

from livekit import rtc

from ...inference import LLM as _InferenceLLM, STT as _InferenceSTT, LLMModels
from ...job import get_job_context
from ...llm import LLM as _LLM
from ...log import logger
from ...stt import STT as _STT
from ...telemetry import trace_types, tracer
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.participant import wait_for_track_publication
from .classifier import (
    AMD_PROMPT,
    HUMAN_SILENCE_THRESHOLD,
    HUMAN_SPEECH_THRESHOLD,
    MACHINE_SILENCE_THRESHOLD,
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

EVALUATED_LLM_MODELS: set[str] = {
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

EVALUATED_STT_MODELS: set[str] = {
    "deepgram/nova-3",
    "assemblyai/universal-streaming-multilingual",
    "cartesia/ink-whisper",
}


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
        session: The :class:`AgentSession` to wire AMD to.
        llm: LLM used for greeting classification. Accepts an :class:`LLM`
            instance or an inference model string (e.g.
            ``"openai/gpt-4.1-mini"``). Omit to fall back to the session's
            own LLM.
        interrupt_on_machine: If ``True`` (default), interrupt any pending
            agent speech immediately when a machine is detected.
        ivr_detection: If ``True`` (default), automatically start IVR
            navigation when a ``machine-ivr`` result is returned.
        participant_identity: If set, only this participant's audio track
            subscription triggers the detection timers. If omitted, the first
            remote audio track wins.
        stt: STT used for transcript generation. Required when the session
            uses no STT (e.g. a realtime model). Omit to reuse the session's
            STT transcripts.
        suppress_compatibility_warning: If ``True``, do not log a warning when
            the resolved STT or LLM is not among the bundled AMD-tested model
            strings. Has no effect on classification behavior.
        detection_options: Optional overrides for timing thresholds and the AMD
            classification prompt (see :class:`DetectionOptions`). When
            omitted, library defaults apply.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        llm: NotGivenOr[LLM | LLMModels | str] = "google/gemini-3.1-flash-lite-preview",
        stt: NotGivenOr[STT | str] = "cartesia/ink-whisper",
        interrupt_on_machine: bool = True,
        ivr_detection: bool = True,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        suppress_compatibility_warning: bool = False,
        detection_options: NotGivenOr[DetectionOptions] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        self._llm_config: NotGivenOr[LLM | LLMModels | str] = llm
        self._session: AgentSession = session
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection
        self._suppress_compatibility_warning = suppress_compatibility_warning
        self._participant_identity: NotGivenOr[str] = participant_identity
        self._stt: NotGivenOr[_STT] = _InferenceSTT(stt) if isinstance(stt, str) else stt

        self._classifier: _AMDClassifier | None = None
        self._result: AMDPredictionEvent | None = None
        self._closed = False
        self._span: trace.Span | None = None

        self._opts: DetectionOptions = (
            {**_DEFAULT_DETECTION_OPTIONS, **detection_options}
            if is_given(detection_options)
            else _DEFAULT_DETECTION_OPTIONS
        )

        if not self._suppress_compatibility_warning:
            _warn_if_not_evaluated(
                self._stt.model if is_given(self._stt) else None,
                EVALUATED_STT_MODELS,
                model_kind="stt",
            )

        self._stt_task: asyncio.Task[None] | None = None
        self._audio_ch: aio.Chan[rtc.AudioFrame] | None = None

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    @property
    def started(self) -> bool:
        return self._classifier is not None and self._classifier.started

    async def execute(self) -> AMDPredictionEvent:
        """Run AMD and return the result.

        While executing, speech playout authorization is locked. Once the
        result is available, authorization is resumed and automatic actions
        (interrupt on machine, ivr detection) are applied based on the
        configured options.
        """
        if self._classifier:
            await self._classifier._verdict_ready.wait()

        if not self._result:
            raise RuntimeError("amd closed before a result was available")

        result = self._result

        if result.is_machine and self._interrupt_on_machine:
            await self._session.interrupt(force=True)

        if result.category == AMDCategory.MACHINE_IVR and self._ivr_detection:
            await self._session._start_ivr_detection(
                transcript=result.transcript,
            )

        # eagerly resume so agent can speak immediately to a human
        if self._session._activity:
            self._session._activity._resume_authorization()

        return result

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
        if self._audio_ch and not self._audio_ch.closed and self._classifier:
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

        if self._stt_task:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                pass
            self._stt_task = None

        if self._classifier:
            self._classifier.off("amd_prediction", self._on_amd_prediction)
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
        self._classifier.on("amd_prediction", self._on_amd_prediction)
        self._closed = False
        self._result = None

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

        # start the classifier first and the timers later when the track is subscribed
        self._classifier.start()
        self._start_span()
        if session._activity:
            session._activity._pause_authorization()

        self._stt_task = asyncio.create_task(self._setup(session), name="amd_setup")

    async def _setup(self, session: AgentSession) -> None:
        if self._closed:
            return
        if not session._room_io:
            logger.warning(
                "session room_io unavailable, starting amd timers immediately as fallback"
            )
            if self._classifier:
                self._classifier.start_timers()
        else:
            await wait_for_track_publication(
                room=session._room_io.room,
                identity=self._participant_identity or None,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
            )
            if not self._closed and self._classifier:
                self._classifier.start_timers()

        if is_given(self._stt) and not self._closed:
            logger.debug("starting amd stt pipeline")
            await self._run_stt()

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
            finally:
                await aio.cancel_and_wait(*tasks)

    def _on_amd_prediction(self, result: AMDPredictionEvent) -> None:
        self._result = result
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

        if not self._suppress_compatibility_warning:
            _warn_if_not_evaluated(
                _llm.model if _llm else None,
                EVALUATED_LLM_MODELS,
                model_kind="llm",
            )

        if _llm:
            return _AMDClassifier(
                _llm,
                human_speech_threshold=self._opts["human_speech_threshold"],
                human_silence_threshold=self._opts["human_silence_threshold"],
                machine_silence_threshold=self._opts["machine_silence_threshold"],
                no_speech_threshold=self._opts["no_speech_threshold"],
                timeout=self._opts["timeout"],
                prompt=self._opts["prompt"],
                source="amd_stt" if is_given(self._stt) else "stt",
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
