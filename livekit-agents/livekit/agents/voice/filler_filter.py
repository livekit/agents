"""
Filler Word Filter Module

This module provides intelligent filtering of filler words during agent speech,
while allowing them when the agent is quiet. It distinguishes between filler-only
utterances and genuine interruptions.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from ..log import logger
from ..tokenize.basic import split_words

if TYPE_CHECKING:
    from .agent_activity import AgentActivity


class FillerWordFilter:
    """
    Filters filler words intelligently based on agent speaking state.
    
    - When agent is speaking: ignores filler-only utterances
    - When agent is quiet: allows all speech including fillers
    - Detects mixed filler + command scenarios and allows them through
    """

    def __init__(
        self,
        ignored_words: list[str] | None = None,
        min_confidence: float = 0.0,
        case_sensitive: bool = False,
    ) -> None:
        """
        Initialize the filler word filter.

        Args:
            ignored_words: List of filler words/phrases to ignore when agent is speaking.
                          Defaults to ['uh', 'umm', 'hmm', 'haan'] if not provided.
            min_confidence: Minimum ASR confidence threshold for filtering (0.0-1.0).
                           Lower confidence transcripts are more likely to be ignored.
            case_sensitive: Whether filler word matching should be case-sensitive.
        """
        if ignored_words is None:
            # Default filler words - can be overridden via environment
            env_words = os.getenv("LIVEKIT_IGNORED_FILLER_WORDS", "")
            if env_words:
                ignored_words = [w.strip() for w in env_words.split(",") if w.strip()]
            else:
                ignored_words = ["uh", "umm", "hmm", "haan"]

        self.ignored_words = ignored_words
        self.min_confidence = min_confidence
        self.case_sensitive = case_sensitive

        # Create regex patterns for matching filler words
        flags = 0 if case_sensitive else re.IGNORECASE
        self._filler_patterns = [
            re.compile(rf"\b{re.escape(word)}\b", flags) for word in ignored_words
        ]

        logger.info(
            "FillerWordFilter initialized",
            extra={
                "ignored_words": ignored_words,
                "min_confidence": min_confidence,
                "case_sensitive": case_sensitive,
            },
        )

    def update_ignored_words(self, words: list[str]) -> None:
        """Dynamically update the list of ignored words."""
        self.ignored_words = words
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._filler_patterns = [
            re.compile(rf"\b{re.escape(word)}\b", flags) for word in words
        ]
        logger.info("FillerWordFilter: updated ignored words", extra={"words": words})

    def _is_filler_only(self, text: str) -> bool:
        """
        Check if the transcript contains only filler words.

        Args:
            text: The transcript text to check.

        Returns:
            True if the text contains only filler words, False otherwise.
        """
        if not text or not text.strip():
            return False

        # Normalize text: remove extra whitespace, convert to lowercase if needed
        normalized = text.strip()
        if not self.case_sensitive:
            normalized = normalized.lower()

        # Split into words for analysis
        words = split_words(normalized, split_character=True)
        if not words:
            return False

        # Check if all words are fillers
        all_fillers = True
        for word in words:
            word_lower = word.lower() if not self.case_sensitive else word
            is_filler = any(
                pattern.search(word_lower) for pattern in self._filler_patterns
            )
            if not is_filler:
                all_fillers = False
                break

        return all_fillers

    def _has_valid_speech(self, text: str) -> bool:
        """
        Check if the transcript contains valid (non-filler) speech.

        Args:
            text: The transcript text to check.

        Returns:
            True if the text contains non-filler words, False otherwise.
        """
        if not text or not text.strip():
            return False

        normalized = text.strip()
        if not self.case_sensitive:
            normalized = normalized.lower()

        words = split_words(normalized, split_character=True)
        if not words:
            return False

        # Check if any word is NOT a filler
        for word in words:
            word_lower = word.lower() if not self.case_sensitive else word
            is_filler = any(
                pattern.search(word_lower) for pattern in self._filler_patterns
            )
            if not is_filler:
                return True  # Found a non-filler word

        return False

    def should_ignore(
        self,
        text: str,
        confidence: float | None = None,
        agent_speaking: bool = False,
    ) -> tuple[bool, str]:
        """
        Determine if a transcript should be ignored based on filler word filtering.

        Args:
            text: The transcript text to evaluate.
            confidence: ASR confidence score (0.0-1.0), if available.
            agent_speaking: Whether the agent is currently speaking.

        Returns:
            Tuple of (should_ignore: bool, reason: str)
            - should_ignore: True if the transcript should be ignored
            - reason: Explanation for the decision (for logging)
        """
        if not text or not text.strip():
            return False, "empty_transcript"

        # Always allow speech when agent is quiet
        if not agent_speaking:
            return False, "agent_not_speaking"

        # Check confidence threshold
        if confidence is not None and confidence < self.min_confidence:
            return True, f"low_confidence_{confidence:.2f}"

        # Check if it's filler-only
        if self._is_filler_only(text):
            return True, "filler_only"

        # Check if it has valid speech (mixed filler + command)
        if self._has_valid_speech(text):
            return False, "contains_valid_speech"

        # Default: don't ignore (edge case handling)
        return False, "unknown"

    def filter_transcript(
        self,
        activity: AgentActivity,
        text: str,
        confidence: float | None = None,
    ) -> tuple[bool, str]:
        """
        Filter a transcript based on agent speaking state and filler words.

        Args:
            activity: The AgentActivity instance to check speaking state.
            text: The transcript text to filter.
            confidence: ASR confidence score, if available.

        Returns:
            Tuple of (should_ignore: bool, reason: str)
        """
        # Check if agent is currently speaking
        agent_speaking = (
            activity._current_speech is not None
            and not activity._current_speech.interrupted
        ) or (activity._session.agent_state == "speaking")

        return self.should_ignore(text, confidence, agent_speaking)

