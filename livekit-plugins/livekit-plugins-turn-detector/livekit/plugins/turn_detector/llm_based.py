from __future__ import annotations

import logging

from livekit.agents import LanguageCode, llm
from livekit.agents.llm import ChatContext

__all__ = ["LLMTurnDetector"]

logger = logging.getLogger("livekit.plugins.turn_detector.llm_based")

_MAX_HISTORY_TURNS = 6
_DEFAULT_TIMEOUT = 1.5
_COMPLETE_PROBABILITY = 0.95
_INCOMPLETE_PROBABILITY = 0.05
_NEUTRAL_PROBABILITY = 0.5

_DEFAULT_INSTRUCTIONS = """\
You are a turn-completion classifier for a voice assistant. Given a transcribed
conversation, decide whether the LAST user message represents a complete thought
that the assistant should respond to, or whether the user is mid-sentence and
likely to continue speaking.

Reply with EXACTLY one token:
- "1" if the user's turn is complete
- "0" if the user appears cut off or still thinking

Do not explain. Do not add punctuation."""


class LLMTurnDetector:
    """Classify end-of-turn using a user-supplied LLM.

    Implements the ``_TurnDetector`` Protocol from
    ``livekit.agents.voice.turn``. Designed as a drop-in alternative to the
    ONNX EOU model for users who would rather spend an LLM call than run a
    dedicated classifier.
    """

    def __init__(
        self,
        llm: llm.LLM,
        *,
        instructions: str | None = None,
        unlikely_threshold: float = _NEUTRAL_PROBABILITY,
        timeout: float = _DEFAULT_TIMEOUT,
        max_history_turns: int = _MAX_HISTORY_TURNS,
    ) -> None:
        self._llm = llm
        self._instructions = instructions or _DEFAULT_INSTRUCTIONS
        self._unlikely_threshold = unlikely_threshold
        self._timeout = timeout
        self._max_history_turns = max_history_turns

    @property
    def provider(self) -> str:
        return "llm"

    @property
    def model(self) -> str:
        return self._llm.model

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None:
        return self._unlikely_threshold

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return True

    async def predict_end_of_turn(
        self, chat_ctx: ChatContext, *, timeout: float | None = None
    ) -> float:
        # Implemented in Task 3.
        raise NotImplementedError
