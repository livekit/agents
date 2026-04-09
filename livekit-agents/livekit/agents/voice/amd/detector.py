from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from opentelemetry import trace

from ...inference import LLM as _InferenceLLM, LLMModels
from ...job import get_job_context
from ...llm import LLM as _LLM
from ...log import logger
from ...telemetry import trace_types, tracer
from .classifier import (
    AMDCategory,
    AMDResult,
    _AMDClassifier,
)

if TYPE_CHECKING:
    from ...llm import LLM
    from ..agent_session import AgentSession


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

    AMD must be wired to an :class:`AgentSession` before classification begins.
    The recommended pattern is the async context manager, which handles pausing
    and resuming speech playout around the detection window::

        async with AMD(session, llm="openai/gpt-4.1-mini") as detector:
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
    """

    def __init__(
        self,
        session: AgentSession | None = None,
        *,
        llm: LLM | LLMModels | str | None = None,
        interrupt_on_machine: bool = True,
        ivr_detection: bool = True,
    ) -> None:
        self._llm_config = llm
        self._session: AgentSession | None = session
        self._classifier: _AMDClassifier | None = None
        self._result: AMDResult | None = None
        self._closed = False
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection
        self._span: trace.Span | None = None

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
            await self._session._start_ivr_detection(transcript=result.transcript)

        # eagerly resume so agent can speak immediately to a human
        if self._session._activity is not None:
            self._session._activity._resume_authorization()

        return result

    async def __aenter__(self) -> AMD:
        if self._session is not None:
            await self._run(self._session)
            # if the session is already running, start classifier and pause
            # authorization immediately (audio may already be flowing)
            if self._classifier is not None and not self._classifier.started:
                self._classifier.start()
                self._start_span()
                if self._session._activity is not None:
                    self._session._activity._pause_authorization()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # region: lifecycle hooks (called by AudioRecognition)

    def _on_first_audio(self) -> None:
        """Start AMD on the first audio frame and pause speech authorization."""
        if self._classifier is None or self._classifier.started:
            return
        self._classifier.start()
        self._start_span()
        if self._session is not None and self._session._activity is not None:
            self._session._activity._pause_authorization()

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

    def _on_amd_result(self, result: AMDResult) -> None:
        self._result = result
        if self._classifier is not None:
            self._classifier.end_input()

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

    @staticmethod
    def _resolve_classifier(
        llm: LLM | LLMModels | str | None,
        session: AgentSession,
    ) -> _AMDClassifier | None:

        if isinstance(llm, str):
            return _AMDClassifier(_InferenceLLM(llm))
        if isinstance(llm, _LLM):
            return _AMDClassifier(llm)

        if (candidate := session.llm) is not None and isinstance(candidate, _LLM):
            return _AMDClassifier(candidate)

        return None

    # endregion
