"""
Filler-Aware Interruption Filter

This module provides intelligent interruption handling that distinguishes
between meaningful user interruptions and filler words/phrases.
"""

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class InterruptDecision:
    """Represents the decision made by the filter."""

    def __init__(
        self,
        action: str,  # "ignore", "interrupt", "pass_through"
        reason: str,
        text: str,
        confidence: float,
    ):
        self.action = action
        self.reason = reason
        self.text = text
        self.confidence = confidence

    def __repr__(self):
        return f"InterruptDecision(action={self.action}, reason={self.reason})"


class FillerAwareInterruptFilter:
    """
    Filters filler words/phrases when the agent is speaking, but allows
    genuine interruptions to pass through.

    The filter distinguishes between:
    - Filler-only speech (e.g., "uh", "umm", "hmm") → ignored when agent speaking
    - Real interruptions (e.g., "stop", "wait") → always interrupt
    - Meaningful speech → interrupt if confidence/word count thresholds met
    """

    def __init__(
        self,
        ignored_words: Optional[list[str]] = None,
        interrupt_keywords: Optional[list[str]] = None,
        min_asr_confidence: float = 0.55,
        interrupt_confidence: float = 0.70,
        min_meaningful_tokens: int = 2,
    ):
        """
        Initialize the filter.

        Args:
            ignored_words: List of filler words to ignore when agent is speaking.
                          Default: ["uh", "umm", "hmm", "haan"]
            interrupt_keywords: Keywords that always trigger interruption.
                               Default: ["stop", "wait", "hold on", "no", "cancel"]
            min_asr_confidence: Minimum ASR confidence to consider filler-only speech.
                               Default: 0.55
            interrupt_confidence: Minimum confidence for non-keyword interruptions.
                                Default: 0.70
            min_meaningful_tokens: Minimum number of meaningful tokens for interruption.
                                  Default: 2
        """
        if ignored_words:
            self._ignored_words = {w.lower().strip() for w in ignored_words}
        else:
            self._ignored_words = {"uh", "umm", "hmm", "haan"}

        if interrupt_keywords:
            self._interrupt_keywords = {kw.lower().strip() for kw in interrupt_keywords}
        else:
            self._interrupt_keywords = {"stop", "wait", "hold on", "no", "cancel", "pause"}

        self._min_asr_confidence = min_asr_confidence
        self._interrupt_confidence = interrupt_confidence
        self._min_meaningful_tokens = min_meaningful_tokens

        self._is_agent_speaking = False
        self._on_valid_interrupt: Optional[Callable[[], None]] = None

    def update_speaking_state(self, is_speaking: bool) -> None:
        """Update the agent's speaking state."""
        if self._is_agent_speaking != is_speaking:
            logger.debug(
                f"Agent speaking state changed: {self._is_agent_speaking} -> {is_speaking}"
            )
            self._is_agent_speaking = is_speaking

    def handle_transcript(
        self, text: str, confidence: float, is_final: bool = True
    ) -> InterruptDecision:
        """
        Process a transcript and decide whether to ignore, interrupt, or pass through.

        Args:
            text: The transcribed text
            confidence: ASR confidence score (0.0-1.0)
            is_final: Whether this is a final transcript

        Returns:
            InterruptDecision with action and reason
        """
        if not text or not text.strip():
            return InterruptDecision(
                action="ignore", reason="empty_transcript", text=text, confidence=confidence
            )

        # Normalize text: lowercase, strip punctuation, collapse spaces
        normalized = self._normalize_text(text)
        tokens = self._tokenize(normalized)

        # If agent is not speaking, always pass through
        if not self._is_agent_speaking:
            logger.debug(
                f"Agent not speaking, passing through: '{text}' (confidence={confidence:.2f})"
            )
            return InterruptDecision(
                action="pass_through",
                reason="agent_not_speaking",
                text=text,
                confidence=confidence,
            )

        # Check for interrupt keywords first (highest priority)
        if self._contains_interrupt_keyword(tokens):
            logger.info(
                f"Valid interrupt detected (keyword): '{text}' (confidence={confidence:.2f})"
            )
            return InterruptDecision(
                action="interrupt",
                reason="valid_interrupt_keyword",
                text=text,
                confidence=confidence,
            )

        # Check if all tokens are filler words
        meaningful_tokens = [t for t in tokens if t not in self._ignored_words]

        # If all tokens are fillers and confidence is low, ignore
        if len(meaningful_tokens) == 0 and confidence < self._min_asr_confidence:
            logger.debug(
                f"Ignored filler (low confidence): '{text}' (confidence={confidence:.2f})"
            )
            return InterruptDecision(
                action="ignore",
                reason="ignored_filler_low_conf",
                text=text,
                confidence=confidence,
            )

        # If meaningful tokens meet threshold or confidence is high, interrupt
        if (
            len(meaningful_tokens) >= self._min_meaningful_tokens
            or confidence >= self._interrupt_confidence
        ):
            logger.info(
                f"Valid interrupt detected (meaningful): '{text}' "
                f"(tokens={len(meaningful_tokens)}, confidence={confidence:.2f})"
            )
            return InterruptDecision(
                action="interrupt",
                reason="valid_interrupt_confidence" if confidence >= self._interrupt_confidence else "valid_interrupt_tokens",
                text=text,
                confidence=confidence,
            )

        # Default: ignore (filler-only with medium confidence)
        logger.debug(
            f"Ignored filler (medium confidence): '{text}' (confidence={confidence:.2f})"
        )
        return InterruptDecision(
            action="ignore",
            reason="ignored_filler_medium_conf",
            text=text,
            confidence=confidence,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove punctuation, collapse spaces."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return [t for t in text.split() if t]

    def _contains_interrupt_keyword(self, tokens: list[str]) -> bool:
        """Check if any token matches an interrupt keyword."""
        token_set = set(tokens)
        # Check for exact matches
        if token_set & self._interrupt_keywords:
            return True
        # Check for multi-word keywords (e.g., "hold on")
        text = " ".join(tokens)
        for keyword in self._interrupt_keywords:
            if " " in keyword and keyword in text:
                return True
        return False

    def update_ignored_words(self, words: list[str]) -> None:
        """Dynamically update the ignored words list (thread-safe for reads)."""
        self._ignored_words = {w.lower().strip() for w in words}
        logger.info(f"Updated ignored words: {self._ignored_words}")

    def update_interrupt_keywords(self, keywords: list[str]) -> None:
        """Dynamically update the interrupt keywords list."""
        self._interrupt_keywords = {kw.lower().strip() for kw in keywords}
        logger.info(f"Updated interrupt keywords: {self._interrupt_keywords}")

    @property
    def is_agent_speaking(self) -> bool:
        """Get current agent speaking state."""
        return self._is_agent_speaking

