from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from ...inference import LLM as _InferenceLLM, LLMModels
from ...llm import LLM as _LLM, RealtimeModel as _RealtimeModel
from .classifier import AMDResult, _AMDClassifier

if TYPE_CHECKING:
    from ...llm import LLM
    from ..agent_activity import AgentActivity
    from ..agent_session import AgentSession


class AMD:
    """Answering-machine detector that owns the full AMD lifecycle.

    Example::

        session = AgentSession(amd=True, llm=..., ...)

        # inside Agent.on_enter:
        result = await self.session.amd.result()
        if result.is_human:
            self.session.amd.resume_authorization()
    """

    def __init__(
        self,
        *,
        llm: bool | LLM | LLMModels | str,
        session: AgentSession,
        activity: AgentActivity | None = None,
    ) -> None:
        self._session = session
        self._classifier: _AMDClassifier | None = self._resolve_classifier(
            llm, session, activity=activity
        )
        self._result: asyncio.Future[AMDResult] | None = (
            asyncio.get_running_loop().create_future() if self._classifier is not None else None
        )
        self._closed = False

        if self._classifier is not None:
            self._classifier.on("amd_result", self._on_amd_result)

    @property
    def enabled(self) -> bool:
        return self._classifier is not None

    @property
    def pending(self) -> bool:
        return self._result is not None and not self._result.done()

    # region: lifecycle methods

    def on_first_audio(self) -> None:
        """Start AMD on the first audio frame and pause speech authorization."""
        if self._classifier is None or self._classifier.started:
            return
        self._classifier.start()
        if self._session._activity is not None:
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
        if self._result is not None and not self._result.done():
            self._result.cancel()

    # endregion
    # region: public API

    async def result(self) -> AMDResult:
        """Wait for the answering-machine detection result and return it.

        Raises:
            RuntimeError: If AMD was not enabled or could not be resolved.
        """
        if self._result is None:
            raise RuntimeError("AMD could not be resolved, please provide a compatible LLM")
        return await self._result

    def resume_authorization(self) -> None:
        """Resume speech playout authorization so queued responses can play."""
        if self._session._activity is not None:
            self._session._activity._resume_authorization()

    # endregion
    # region: internal methods

    def _on_amd_result(self, result: AMDResult) -> None:
        if self._result is not None and not self._result.done():
            with contextlib.suppress(asyncio.InvalidStateError):
                self._result.set_result(result)
        task = asyncio.create_task(self.aclose())
        task.add_done_callback(lambda _: None)

    @staticmethod
    def _resolve_classifier(
        llm: bool | LLM | LLMModels | str,
        session: AgentSession,
        *,
        activity: AgentActivity | None = None,
    ) -> _AMDClassifier | None:

        if not llm:
            return None
        if isinstance(llm, str):
            return _AMDClassifier(_InferenceLLM(llm))
        if isinstance(llm, _LLM):
            return _AMDClassifier(llm)

        candidates = [session.llm]
        if activity is not None:
            candidates.insert(0, activity.llm)

        for candidate in candidates:
            if candidate is None or isinstance(candidate, _RealtimeModel):
                continue
            if isinstance(candidate, _LLM):
                return _AMDClassifier(candidate)

        return None

    # endregion
