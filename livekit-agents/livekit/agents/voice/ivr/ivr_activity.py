from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from livekit.rtc import EventEmitter

from ...log import logger
from ...utils.aio.debounce import Debounced

if TYPE_CHECKING:
    from ..agent import Agent
    from ..agent_session import AgentSession
    from ..events import AgentStateChangedEvent, UserInputTranscribedEvent, UserStateChangedEvent


DEFAULT_SILENCE_REMINDER_MESSAGE = (
    "<system_notification>"
    "Now that the phone call is become silent for a while, say something or perform some action to proceed with the call. "
    "</system_notification>"
)

DEFAULT_LOOP_DETECTED_MESSAGE = (
    "<system_notification>"
    "Speech loop has been detected from the automated IVR system, say something or perform some action to proceed with the call. "
    "</system_notification>"
)

LoopDetectorEventTypes = Literal["loop_detected"]
SilenceDetectorEventTypes = Literal["silence_detected"]


class IVRActivity:
    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        self._session = session
        self._silence_detector = SilenceDetector(session, max_silence_duration=max_silence_duration)
        self._loop_detector = TfidfLoopDetector(session)

    async def start(self) -> None:
        self._silence_detector.on("silence_detected", self._on_silence_detected)
        self._loop_detector.on("loop_detected", self._on_loop_detected)

        await self._silence_detector.start()
        await self._loop_detector.start()

    async def update_agent(self, agent: Agent) -> None:
        from ...beta.tools.send_dtmf import send_dtmf_events

        agent._tools.append(send_dtmf_events)

    def _on_silence_detected(self, _) -> None:
        logger.info("IVRActivity: silence detected; sending notification")
        self._session.generate_reply(user_input=DEFAULT_SILENCE_REMINDER_MESSAGE)

    def _on_loop_detected(self, _) -> None:
        logger.info("IVRActivity: speech loop detected; sending notification")
        self._loop_detector.reset()
        self._session.generate_reply(
            user_input=DEFAULT_LOOP_DETECTED_MESSAGE, allow_interruptions=False
        )

    async def aclose(self) -> None:
        self._silence_detector.off("silence_detected", self._on_silence_detected)
        self._loop_detector.off("loop_detected", self._on_loop_detected)

        await self._silence_detector.aclose()
        await self._loop_detector.aclose()


class TfidfLoopDetector(EventEmitter[LoopDetectorEventTypes]):
    """TF-IDF based loop detector.

    This detector uses TF-IDF to detect loops in the user's input by comparing
    the similarity of the last N - 1 chunks of transcribed text to the last chunk.

    Args:
        session: The agent session.
        window_size: The number of chunks to compare. Default ``20``.
        similarity_threshold: The similarity threshold for a chunk to be considered similar to the last chunk. Default ``0.9``.
        consecutive_threshold: The number of consecutive chunks that must be similar to trigger a loop detection. Default ``3``.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        window_size: int = 20,
        similarity_threshold: float = 0.85,
        consecutive_threshold: int = 3,
    ) -> None:
        super().__init__()

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if consecutive_threshold <= 0:
            raise ValueError("consecutive_threshold must be greater than 0")

        self._session = session
        self._window_size = window_size
        self._similarity_threshold = similarity_threshold
        self._consecutive_threshold = consecutive_threshold
        self._vectorizer = TfidfVectorizer()
        self._transcribed_chunks: list[str] = []
        self._num_consecutive_similar_chunks = 0

    @property
    def loop_detected(self):
        return self._num_consecutive_similar_chunks >= self._consecutive_threshold

    async def start(self) -> None:
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)

    def reset(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._transcribed_chunks = []
        self._num_consecutive_similar_chunks = 0

    async def aclose(self) -> None:
        self._session.off("user_input_transcribed", self._on_user_input_transcribed)

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final:
            return

        self._transcribed_chunks.append(ev.transcript)
        if len(self._transcribed_chunks) > self._window_size:
            self._transcribed_chunks = self._transcribed_chunks[-self._window_size :]

        # NOTE: currently this is O(n^2) in the number of chunks, let's figure out a more efficient
        # way if this become a bottleneck later.
        doc_matrix = self._vectorizer.fit_transform(self._transcribed_chunks)
        doc_similarity = cosine_similarity(doc_matrix)
        last_chunk_similarity = doc_similarity[-1][:-1]

        if np.max(last_chunk_similarity) > self._similarity_threshold:
            self._num_consecutive_similar_chunks += 1
        else:
            self._num_consecutive_similar_chunks = 0

        if self.loop_detected:
            self.emit("loop_detected", None)


class SilenceDetector(EventEmitter[SilenceDetectorEventTypes]):
    """Silence detector.

    This detector checks for silence in the user / agent interaction.

    Args:
        session: The agent session.
        max_silence_duration: The maximum duration of silence to detect in seconds. Default ``5.0`` seconds.
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        super().__init__()
        self._session = session
        self._max_silence_duration = max_silence_duration
        self._current_user_state: Optional[str] = None  # noqa: UP007
        self._current_agent_state: Optional[str] = None  # noqa: UP007
        self._debounced_emit = Debounced(self._emit_silence_detected, self._max_silence_duration)

    async def start(self) -> None:
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("agent_state_changed", self._on_agent_state_changed)

    async def aclose(self) -> None:
        self._debounced_emit.cancel()
        self._session.off("user_state_changed", self._on_user_state_changed)
        self._session.off("agent_state_changed", self._on_agent_state_changed)

    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        self._current_user_state = ev.new_state
        self._schedule_check()

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        self._current_agent_state = ev.new_state
        self._schedule_check()

    def _schedule_check(self) -> None:
        if self._current_user_state == self._current_agent_state == "listening":
            logger.info(
                "SilenceDetector: user_state=%s, agent_state=%s, scheduling silence check",
                self._current_user_state,
                self._current_agent_state,
            )
            self._debounced_emit.schedule()
        else:
            logger.info(
                "SilenceDetector: user_state=%s, agent_state=%s, canceling silence check",
                self._current_user_state,
                self._current_agent_state,
            )
            self._debounced_emit.cancel()

    async def _emit_silence_detected(self) -> None:
        logger.info("SilenceDetector: emitting silence_detected event")
        self.emit("silence_detected", None)
