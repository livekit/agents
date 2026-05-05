from __future__ import annotations

import asyncio
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
        user_messages = [m for m in chat_ctx.messages() if m.role == "user"]
        if not user_messages:
            return 1.0

        prompt_ctx = self._build_prompt_ctx(chat_ctx)
        effective_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await asyncio.wait_for(
                self._llm.chat(chat_ctx=prompt_ctx).collect(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("LLMTurnDetector: classifier timed out after %.2fs", effective_timeout)
            return _NEUTRAL_PROBABILITY
        except Exception:
            logger.warning("LLMTurnDetector: classifier call failed", exc_info=True)
            return _NEUTRAL_PROBABILITY

        return self._parse_probability(response.text or "")

    def _build_prompt_ctx(self, chat_ctx: ChatContext) -> ChatContext:
        messages = chat_ctx.messages()[-self._max_history_turns :]
        lines: list[str] = []
        for i, msg in enumerate(messages):
            text = msg.text_content or ""
            marker = "[CURRENT] " if i == len(messages) - 1 and msg.role == "user" else ""
            lines.append(f"{marker}{msg.role}: {text}")
        rendered = "\n".join(lines)

        prompt_ctx = ChatContext.empty()
        prompt_ctx.add_message(role="system", content=self._instructions)
        prompt_ctx.add_message(role="user", content=rendered)
        return prompt_ctx

    def _parse_probability(self, content: str) -> float:
        stripped = content.strip()
        if not stripped:
            logger.warning("LLMTurnDetector: empty response")
            return _NEUTRAL_PROBABILITY
        first = stripped[0]
        if first == "1":
            return _COMPLETE_PROBABILITY
        if first == "0":
            return _INCOMPLETE_PROBABILITY
        logger.warning("LLMTurnDetector: unexpected response token %r", stripped[:16])
        return _NEUTRAL_PROBABILITY
