from __future__ import annotations

import asyncio
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

    Example::

        amd = AMD("openai/gpt-5-mini")

        # wire AMD to the session before starting it
        await amd.start(session)
        await session.start(agent=MyAgent(), room=ctx.room)

        # inside Agent.on_enter:
        result = await amd.result()
        if result.is_human:
            amd.stop(abort_generation=False)
        else:
            amd.stop(abort_generation=True)
    """

    def __init__(self, llm: LLM | LLMModels | str | None = None) -> None:
        self._llm_config = llm
        self._session: AgentSession | None = None
        self._classifier: _AMDClassifier | None = None
        self._result: AMDResult | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._classifier is not None and self._result is None

    # region: public methods
    async def start(self, session: AgentSession) -> None:
        """Wire AMD to the given session.

        Must be called before ``session.start()`` so that
        :class:`AudioRecognition` picks up the AMD instance for lifecycle hooks.
        """
        self._start_internal(session)

    def stop(self, *, abort_generation: bool = False) -> None:
        """Stop AMD and resume speech playout authorization.

        Args:
            abort_generation: When ``True``, cancel all pending/current
                generation via ``session.interrupt(force=True)`` before
                resuming authorization.  When ``False``, any queued
                responses will play out normally.
        """
        if self._session is None:
            return

        if abort_generation:
            self._session.interrupt(force=True)

        if self._session._activity is not None:
            self._session._activity._resume_authorization()

    async def result(self) -> AMDResult:
        """Wait for the answering-machine detection result and return it.

        Raises:
            RuntimeError: If AMD was not enabled or could not be resolved, or
                if the detector was closed before producing a result.
        """
        if self._classifier is None and self._result is None:
            raise RuntimeError("AMD could not be resolved, please provide a compatible LLM")
        if self._classifier is not None:
            await self._classifier._verdict_ready.wait()
        if self._result is None:
            raise RuntimeError("AMD was closed before a result was available")
        return self._result

    # endregion

    # region: lifecycle hooks

    def on_first_audio(self) -> None:
        """Start AMD on the first audio frame and pause speech authorization."""
        if self._classifier is None or self._classifier.started:
            return
        self._classifier.start()
        if self._session is not None and self._session._activity is not None:
            self._session._activity._pause_authorization()

    def on_user_speech_started(self) -> None:
        if self._classifier is not None:
            self._classifier.on_user_speech_started()

    def on_user_speech_ended(self, silence_duration: float) -> None:
        if self._classifier is not None:
            self._classifier.on_user_speech_ended(silence_duration)

    def on_transcript(self, text: str) -> None:
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
            logger.warning("ivr_detection will be disabled when amd is enabled")
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
