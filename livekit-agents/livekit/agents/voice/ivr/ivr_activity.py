from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from livekit.agents import llm

from ...log import logger
from ...utils.aio.debounce import Debounced

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..events import AgentStateChangedEvent, UserInputTranscribedEvent, UserStateChangedEvent


class IVRActivity:
    def __init__(
        self,
        session: AgentSession,
        *,
        max_silence_duration: float = 5.0,
    ) -> None:
        self._session = session
        self._max_silence_duration = max_silence_duration
        self._loop_detector = TfidfLoopDetector()

        self._current_user_state: str | None = None  # noqa: UP007
        self._current_agent_state: str | None = None  # noqa: UP007
        self._debounced_silence = Debounced(self._on_silence_detected, max_silence_duration)
        self._last_should_schedule_check: bool | None = None

    async def start(self) -> None:
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)

    @property
    def tools(self) -> list[llm.FunctionTool | llm.RawFunctionTool]:
        from ...beta.tools.send_dtmf import send_dtmf_events

        return [send_dtmf_events]

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final:
            return

        self._loop_detector.add_chunk(ev.transcript)

        if self._loop_detector.check_loop_detection():
            logger.debug("IVRActivity: speech loop detected; sending notification")

            self._session.generate_reply(allow_interruptions=False)
            self._loop_detector.reset()

    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        self._current_user_state = ev.new_state
        self._schedule_silence_check()

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        self._current_agent_state = ev.new_state
        self._schedule_silence_check()

    def _schedule_silence_check(self) -> None:
        should_schedule = self._should_schedule_check()
        if should_schedule:
            if self._last_should_schedule_check:
                return

            self._debounced_silence.schedule()
        else:
            self._debounced_silence.cancel()

        self._last_should_schedule_check = should_schedule

    def _should_schedule_check(self) -> bool:
        is_user_silent = self._current_user_state in ["listening", "away"]
        is_agent_silent = self._current_agent_state in ["idle", "listening"]
        return is_user_silent and is_agent_silent

    async def _on_silence_detected(self) -> None:
        logger.debug("IVRActivity: silence detected; sending notification")
        self._session.generate_reply()

    async def aclose(self) -> None:
        self._debounced_silence.cancel()
        self._session.off("user_state_changed", self._on_user_state_changed)
        self._session.off("agent_state_changed", self._on_agent_state_changed)
        self._session.off("user_input_transcribed", self._on_user_input_transcribed)


class TfidfLoopDetector:
    """TF-IDF based loop detector.

    This detector uses TF-IDF to detect loops in the user's input by comparing
    the similarity of the last N - 1 chunks of transcribed text to the last chunk.

    Args:
        window_size: The number of chunks to compare. Default ``20``.
        similarity_threshold: The similarity threshold for a chunk to be considered similar to the last chunk. Default ``0.85``.
        consecutive_threshold: The number of consecutive chunks that must be similar to trigger a loop detection. Default ``3``.
    """

    def __init__(
        self,
        window_size: int = 20,
        similarity_threshold: float = 0.85,
        consecutive_threshold: int = 3,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if consecutive_threshold <= 0:
            raise ValueError("consecutive_threshold must be greater than 0")

        self._window_size = window_size
        self._similarity_threshold = similarity_threshold
        self._consecutive_threshold = consecutive_threshold
        self._transcribed_chunks: list[str] = []
        self._num_consecutive_similar_chunks = 0

    def reset(self) -> None:
        self._transcribed_chunks = []
        self._num_consecutive_similar_chunks = 0

    def add_chunk(self, chunk: str) -> None:
        self._transcribed_chunks.append(chunk)
        if len(self._transcribed_chunks) > self._window_size:
            self._transcribed_chunks = self._transcribed_chunks[-self._window_size :]

    def check_loop_detection(self) -> bool:
        # Need at least two chunks to compute similarity against the last chunk
        if len(self._transcribed_chunks) < 2:
            return False

        doc_matrix = self._tfidf_matrix(self._transcribed_chunks)
        last_chunk_similarity = doc_matrix[:-1] @ doc_matrix[-1]

        if (
            last_chunk_similarity.size > 0
            and np.max(last_chunk_similarity) > self._similarity_threshold
        ):
            self._num_consecutive_similar_chunks += 1
        else:
            self._num_consecutive_similar_chunks = 0

        return self._num_consecutive_similar_chunks >= self._consecutive_threshold

    @staticmethod
    def _tfidf_matrix(chunks: list[str]) -> np.ndarray:
        """Build an L2-normalized TF-IDF matrix for the transcribed chunks."""

        tokenized_chunks = [_TOKEN_PATTERN.findall(chunk.lower()) for chunk in chunks]
        vocabulary = sorted({token for tokens in tokenized_chunks for token in tokens})
        if not vocabulary:
            return np.zeros((len(chunks), 0))

        token_indexes = {token: index for index, token in enumerate(vocabulary)}
        term_frequency = np.zeros((len(chunks), len(vocabulary)))
        for chunk_index, tokens in enumerate(tokenized_chunks):
            for token in tokens:
                term_frequency[chunk_index, token_indexes[token]] += 1

        document_frequency = np.count_nonzero(term_frequency, axis=0)
        inverse_document_frequency = np.log((1 + len(chunks)) / (1 + document_frequency)) + 1
        tfidf = term_frequency * inverse_document_frequency
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        normalized_tfidf: np.ndarray = np.divide(
            tfidf, norms, out=np.zeros_like(tfidf), where=norms != 0
        )
        return normalized_tfidf
