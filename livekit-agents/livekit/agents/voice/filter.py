from __future__ import annotations

import logging
import os
from typing import Set

logger = logging.getLogger(__name__)


class InterruptionFilter:
    """
    Filters interruptions based on what the user said and whether the agent is speaking.
    
    Ignores backchanneling words ("yeah", "ok", "hmm") when agent is speaking.
    Allows all input when agent is silent.
    """

    # Default words that should be ignored when agent is speaking
    DEFAULT_IGNORE_WORDS: Set[str] = {
        "yeah",
        "ok",
        "okay",
        "hmm",
        "mhm",
        "mm-hmm",
        "uh-huh",
        "right",
        "aha",
        "ah",
        "oh",
        "sure",
        "yep",
        "yup",
        "gotcha",
        "got it",
        "alright",
        "cool",
    }

    def __init__(
        self,
        ignore_words: list[str] | None = None,
        enabled: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        """
        Initialize the interruption filter.
        Args:
            ignore_words: List of words/phrases to ignore when agent is speaking.
                         If None, uses DEFAULT_IGNORE_WORDS.
            enabled: Whether the filter is enabled. If False, all interruptions are allowed.
            case_sensitive: Whether word matching should be case-sensitive.
        """
        self._enabled = enabled
        self._case_sensitive = case_sensitive

        # Load ignore words from parameter or environment variable or default
        if ignore_words is not None:
            self._ignore_words = set(ignore_words)
        else:
            env_words = os.getenv("LIVEKIT_INTERRUPTION_IGNORE_WORDS")
            if env_words:
                self._ignore_words = set(w.strip() for w in env_words.split(","))
                logger.info(
                    f"Loaded {len(self._ignore_words)} ignore words from environment variable"
                )
            else:
                self._ignore_words = self.DEFAULT_IGNORE_WORDS.copy()

        if not self._case_sensitive:
            # Normalize to lowercase for case-insensitive matching
            self._ignore_words = {word.lower() for word in self._ignore_words}

        logger.info(
            f"InterruptionFilter initialized with {len(self._ignore_words)} ignore words, "
            f"enabled={self._enabled}, case_sensitive={self._case_sensitive}"
        )

    @property
    def enabled(self) -> bool:
        """Whether the filter is enabled."""
        return self._enabled

    @property
    def ignore_words(self) -> Set[str]:
        """Set of words that are ignored when agent is speaking."""
        return self._ignore_words.copy()

    def should_ignore_interruption(
        self,
        transcribed_text: str,
        agent_is_speaking: bool,
    ) -> bool:
        """
        Determine if an interruption should be ignored.
        Args:
            transcribed_text: The transcribed text from the user's speech.
            agent_is_speaking: Whether the agent is currently speaking.
        Returns:
            True if the interruption should be ignored (agent continues speaking),
            False if the interruption should be processed (agent stops).
        Logic:
            - If filter is disabled: Always return False (allow all interruptions)
            - If agent is NOT speaking: Always return False (process all input)
            - If agent IS speaking:
                - If ALL words are in ignore list: Return True (ignore)
                - If ANY word is NOT in ignore list: Return False (allow interruption)
        """
        # If filter is disabled, allow all interruptions
        if not self._enabled:
            return False

        # If agent is not speaking, process all user input normally
        if not agent_is_speaking:
            return False

        # Agent is speaking - check if this is backchanneling
        return self._is_backchanneling(transcribed_text)

    def _is_backchanneling(self, text: str) -> bool:
        """
        Check if the given text is purely backchanneling (should be ignored).
        Args:
            text: The transcribed text to analyze.
        Returns:
            True if the text is backchanneling (all words in ignore list),
            False if it contains any command or meaningful content.
        """
        if not text or not text.strip():
            # Empty text is not backchanneling
            return False

        # Normalize the text
        normalized_text = text.strip()
        if not self._case_sensitive:
            normalized_text = normalized_text.lower()

        # Remove common punctuation
        normalized_text = normalized_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")

        # Split into words
        words = normalized_text.split()

        if not words:
            return False

        # Check if ALL words are in the ignore list
        # If even one word is NOT in the ignore list, it's not pure backchanneling
        for word in words:
            if word not in self._ignore_words:
                # Found a word that's not in ignore list - this is a real interruption
                logger.debug(
                    f"Detected real interruption: '{text}' contains non-backchannel word '{word}'"
                )
                return False

        # All words are in the ignore list - this is backchanneling
        logger.debug(f"Detected backchanneling: '{text}' - ignoring interruption")
        return True

    def add_ignore_word(self, word: str) -> None:
        """
        Add a word to the ignore list.
        Args:
            word: The word to add to the ignore list.
        """
        normalized_word = word if self._case_sensitive else word.lower()
        self._ignore_words.add(normalized_word)
        logger.debug(f"Added '{word}' to ignore list")

    def remove_ignore_word(self, word: str) -> None:
        """
        Remove a word from the ignore list.
        Args:
            word: The word to remove from the ignore list.
        """
        normalized_word = word if self._case_sensitive else word.lower()
        self._ignore_words.discard(normalized_word)
        logger.debug(f"Removed '{word}' from ignore list")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the filter.
        Args:
            enabled: Whether the filter should be enabled.
        """
        self._enabled = enabled
        logger.info(f"InterruptionFilter {'enabled' if enabled else 'disabled'}")

    def __repr__(self) -> str:
        return (
            f"InterruptionFilter(enabled={self._enabled}, "
            f"ignore_words_count={len(self._ignore_words)}, "
            f"case_sensitive={self._case_sensitive})"
        )
