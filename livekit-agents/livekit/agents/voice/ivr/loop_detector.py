from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from livekit.rtc import EventEmitter

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..events import UserInputTranscribedEvent


EventTypes = Literal["loop_detected"]


class BaseLoopDetector(ABC, EventEmitter[EventTypes]):
    def __init__(self, session: AgentSession) -> None:
        super().__init__()
        self._session = session

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class TfidfLoopDetector(BaseLoopDetector):
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
        similarity_threshold: float = 0.9,
        consecutive_threshold: int = 3,
    ) -> None:
        super().__init__(session)

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if consecutive_threshold <= 0:
            raise ValueError("consecutive_threshold must be greater than 0")

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

        if np.max(doc_similarity[-1]) > self._similarity_threshold:
            self._num_consecutive_similar_chunks += 1
        else:
            self._num_consecutive_similar_chunks = 0

        if self.loop_detected:
            self.emit("loop_detected", None)
