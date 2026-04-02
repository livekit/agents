from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING

from ...inference import LLM as _InferenceLLM, LLMModels
from ...llm import LLM as _LLM
from ...log import logger
from .classifier import AMDResult, _AMDClassifier

if TYPE_CHECKING:
    from ...llm import LLM
    from ..agent_session import AgentSession


class AMD:
    """Answering-machine detector that owns the full AMD lifecycle.

    Can be used as an async context manager or with explicit ``start()``::

        # Pattern 1: context manager
        async with AMD(session, llm="openai/gpt-5-mini") as amd:
            result = await amd.execute()
            if result.is_human:
                ...

        # Pattern 2: explicit start
        amd = AMD(llm="openai/gpt-5-mini")
        await amd.start(session)
        result = await amd.execute()
    """

    def __init__(
        self,
        session: AgentSession | None = None,
        *,
        llm: LLM | LLMModels | str | None = None,
        interrupt_on_machine: bool = True,
        start_ivr_on_dtmf: bool = True,
    ) -> None:
        self._llm_config = llm
        self._session: AgentSession | None = session
        self._classifier: _AMDClassifier | None = None
        self._result: AMDResult | None = None
        self._closed = False
        self._interrupt_on_machine = interrupt_on_machine
        self._start_ivr_on_dtmf = start_ivr_on_dtmf

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    # region: public API

    async def start(self, session: AgentSession) -> None:
        """Wire AMD to the given session.

        Must be called before ``session.start()`` so that
        :class:`AudioRecognition` picks up the AMD instance for lifecycle hooks.
        """
        self._session = session
        self._start_internal(session)

    async def execute(self) -> AMDResult:
        """Run AMD detection and return the result.

        While executing, speech playout authorization is locked. Once the
        result is available, authorization is resumed and automatic actions
        (interrupt on machine, start IVR on DTMF) are applied based on the
        configured options.
        """
        if self._session is None:
            raise RuntimeError("AMD is not wired to a session, call start() first")
        if self._classifier is None and self._result is None:
            raise RuntimeError("AMD could not be resolved, please provide a compatible LLM")

        if self._classifier is not None:
            await self._classifier._verdict_ready.wait()
        if self._result is None:
            raise RuntimeError("AMD was closed before a result was available")

        result = self._result

        if result.is_machine and self._interrupt_on_machine:
            self._session.interrupt(force=True)

        if result.category == "machine-dtmf" and self._start_ivr_on_dtmf:
            await self._session._start_ivr_detection(transcript=result.transcript)

        if self._session._activity is not None:
            self._session._activity._resume_authorization()

        return result

    async def __aenter__(self) -> AMD:
        if self._session is not None:
            self._start_internal(self._session)
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
        if exc_val is not None and self._session is not None:
            # on exception, ensure authorization is resumed so the session isn't stuck
            if self._session._activity is not None:
                self._session._activity._resume_authorization()
        await self.aclose()

    # endregion
    # region: lifecycle hooks (called by AudioRecognition)

    def _on_first_audio(self) -> None:
        """Start AMD on the first audio frame and pause speech authorization."""
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
            self._classifier.off("amd_result", self._on_amd_result)
            await self._classifier.close()
            self._classifier = None

    # endregion
    # region: internal methods

    def _start_internal(self, session: AgentSession) -> None:
        self._session = session
        self._classifier = self._resolve_classifier(self._llm_config, session)
        if self._classifier is None:
            raise ValueError(
                "amd classifier could not be resolved, please provide a compatible model"
            )
        self._classifier.on("amd_result", self._on_amd_result)

        if session.options.ivr_detection:
            logger.warning("ivr_detection will be disabled when AMD is enabled")
            session.options.ivr_detection = False

        session._amd = self

    def _on_amd_result(self, result: AMDResult) -> None:
        self._result = result
        task = asyncio.create_task(self.aclose())
        task.add_done_callback(lambda _: None)

    @staticmethod
    def _resolve_classifier(
        llm: LLM | LLMModels | str | None,
        session: AgentSession,
    ) -> _AMDClassifier | None:
        if isinstance(llm, str):
            return _AMDClassifier(_InferenceLLM(llm))
        if isinstance(llm, _LLM):
            return _AMDClassifier(llm)

        # llm is None — fall back to the session's LLM (skip RealtimeModel)
        candidate = session.llm
        if candidate is not None and isinstance(candidate, _LLM):
            return _AMDClassifier(candidate)

        return None

    # endregion
