from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING

from ...inference import LLM as _InferenceLLM, LLMModels
from ...llm import LLM as _LLM
from ...log import logger
from .classifier import (
    MachineDetectionCategory,
    MachineDetectionResult,
    _MachineDetectionClassifier,
)

if TYPE_CHECKING:
    from ...llm import LLM
    from ..agent_session import AgentSession


class MachineDetector:
    """Detects whether an outbound call is answered by a human or a machine.

    Listens to the call greeting and uses an LLM to classify it into one of
    the following categories:

    - ``human``: a real person answered.
    - ``machine-ivr``: an IVR / DTMF menu prompt was detected.
    - ``machine-vm``: a voicemail greeting where leaving a message is possible.
    - ``machine-unavailable``: the mailbox is full or not set up; leaving a message is not possible.
    - ``uncertain``: the transcript is ambiguous and could not be classified.

    MachineDetector must be wired to an :class:`AgentSession` before classification begins.
    The recommended pattern is the async context manager, which handles pausing
    and resuming speech playout around the detection window::

        async with MachineDetector(session, llm="openai/gpt-4.1-mini") as detector:
            result = await detector.execute()

    Args:
        session: The :class:`AgentSession` to wire MachineDetector to. Can also be passed
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
        self._classifier: _MachineDetectionClassifier | None = None
        self._result: MachineDetectionResult | None = None
        self._closed = False
        self._interrupt_on_machine = interrupt_on_machine
        self._ivr_detection = ivr_detection

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    @property
    def started(self) -> bool:
        return self._classifier is not None and self._classifier.started

    def start(self, session: AgentSession) -> None:
        """Wire MachineDetector to the given session."""
        self._session = session
        self._run(session)

    async def execute(self) -> MachineDetectionResult:
        """Run MachineDetector and return the result.

        While executing, speech playout authorization is locked. Once the
        result is available, authorization is resumed and automatic actions
        (interrupt on machine, ivr detection) are applied based on the
        configured options.
        """
        if self._session is None:
            raise RuntimeError("MachineDetector is not wired to a session, call start() first")

        if self._classifier is not None:
            await self._classifier._verdict_ready.wait()
        if self._result is None:
            raise RuntimeError("MachineDetector was closed before a result was available")

        result = self._result

        if result.is_machine and self._interrupt_on_machine:
            await self._session.interrupt(force=True)

        if result.category == MachineDetectionCategory.MACHINE_IVR and self._ivr_detection:
            await self._session._start_ivr_detection(transcript=result.transcript)

        if self._session._activity is not None:
            self._session._activity._resume_authorization()

        return result

    async def __aenter__(self) -> MachineDetector:
        if self._session is not None:
            self._run(self._session)
            # if the session is already running, start classifier and pause
            # authorization immediately (audio may already be flowing)
            if self._classifier is not None and not self._classifier.started:
                self._classifier.start()
                if self._session._activity is not None:
                    self._session._activity._pause_authorization()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # on exit, resume authorization
        if self._session is not None and self._session._activity is not None:
            self._session._activity._resume_authorization()

        await self.aclose()

    # region: lifecycle hooks (called by AudioRecognition)

    def _on_first_audio(self) -> None:
        """Start MachineDetector on the first audio frame and pause speech authorization."""
        if self._classifier is None or self._classifier.started:
            return
        self._classifier.start()
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
            self._classifier.off("machine_detection_result", self._on_machine_detection_result)
            await self._classifier.close()
            self._classifier = None

        if self._session is not None and self._session._activity is not None:
            self._session._activity._resume_authorization()

    # endregion

    # region: internal methods

    def _run(self, session: AgentSession) -> None:
        if self._classifier is not None:
            logger.warning("machine detector already running, skipping")
            return

        self._session = session
        self._classifier = self._resolve_classifier(self._llm_config, session)
        if self._classifier is None:
            raise ValueError(
                "machine detection classifier could not be resolved, please provide a compatible model"
            )
        self._classifier.on("machine_detection_result", self._on_machine_detection_result)
        self._closed = False
        self._result = None

        if session.options.ivr_detection:
            logger.warning(
                "session level ivr_detection will be disabled when machine detection is used"
            )
            session.options.ivr_detection = False

        session._machine_detector = self

    def _on_machine_detection_result(self, result: MachineDetectionResult) -> None:
        self._result = result
        task = asyncio.create_task(self.aclose())

        def _done_callback(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and (exception := task.exception()) is not None:
                logger.error("error closing machine detector: %s", exception)

        task.add_done_callback(_done_callback)

    @staticmethod
    def _resolve_classifier(
        llm: LLM | LLMModels | str | None,
        session: AgentSession,
    ) -> _MachineDetectionClassifier | None:

        if isinstance(llm, str):
            return _MachineDetectionClassifier(_InferenceLLM(llm))
        if isinstance(llm, _LLM):
            return _MachineDetectionClassifier(llm)

        if (candidate := session.llm) is not None and isinstance(candidate, _LLM):
            return _MachineDetectionClassifier(candidate)

        return None

    # endregion
