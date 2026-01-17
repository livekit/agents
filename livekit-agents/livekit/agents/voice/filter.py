from __future__ import annotations

import logging
import os
import string
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class InterruptionFilter:
    """
    Filters interruptions based on what the user said and whether the agent is speaking.

    Ignores backchanneling words ("yeah", "ok", "hmm") when agent is speaking.
    Allows all input when agent is silent.
    """

    # Default words that should be ignored when agent is speaking
    DEFAULT_IGNORE_WORDS: set[str] = {
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
        "got",
        "it",
        "alright",
        "cool",
    }

    def __init__(
        self,
        ignore_words: list[str] | None = None,
        enabled: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the InterruptionFilter.

        Args:
            ignore_words: Custom list of words to ignore. If None, loads from
                LIVEKIT_INTERRUPTION_IGNORE_WORDS environment variable or
                DEFAULT_IGNORE_WORDS.
            enabled: Whether the interruption filter is active. Defaults to True.
            case_sensitive: Whether word matching is case-sensitive. Defaults to False.
        """
        self._enabled = enabled
        self._case_sensitive = case_sensitive

        # Load ignore words from parameter, environment variable, or defaults
        if ignore_words is not None:
            raw_words = set(ignore_words)
        else:
            env_words = os.getenv("LIVEKIT_INTERRUPTION_IGNORE_WORDS")
            if env_words:
                raw_words = {w.strip() for w in env_words.split(",")}
                logger.info(
                    "Loaded %d ignore words from environment variable",
                    len(raw_words),
                )
            else:
                raw_words = self.DEFAULT_IGNORE_WORDS.copy()

        # Normalize ignore words once
        self._ignore_words = {
            self._normalize_text(word) for word in raw_words if word.strip()
        }

        logger.info(
            "InterruptionFilter initialized with %d ignore words, enabled=%s, case_sensitive=%s",
            len(self._ignore_words),
            self._enabled,
            self._case_sensitive,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        normalized = text.strip()
        if not self._case_sensitive:
            normalized = normalized.lower()

        translator: Dict[int, Optional[int]] = str.maketrans(
            "", "", string.punctuation
        )
        return normalized.translate(translator)

    @property
    def enabled(self) -> bool:
        """Whether the filter is enabled."""
        return self._enabled

    @property
    def ignore_words(self) -> set[str]:
        """Set of words that are ignored when agent is speaking."""
        return self._ignore_words.copy()

    def should_ignore_interruption(
        self,
        transcribed_text: str,
        agent_is_speaking: bool,
    ) -> bool:
        if not self._enabled:
            return False

        if not agent_is_speaking:
            return False

        return self._is_backchanneling(transcribed_text)

    def _is_backchanneling(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        normalized_text = self._normalize_text(text)

        words = normalized_text.split()
        if not words:
            return False

        normalized_phrase = " ".join(words)

        # Phrase-level match (e.g. "got it")
        if normalized_phrase in self._ignore_words:
            logger.debug(
                "Detected backchanneling phrase: '%s' - ignoring interruption",
                text,
            )
            return True

        # Word-by-word validation
        for word in words:
            if word not in self._ignore_words:
                logger.debug(
                    "Detected real interruption: '%s' contains non-backchannel word '%s'",
                    text,
                    word,
                )
                return False

        logger.debug("Detected backchanneling: '%s' - ignoring interruption", text)
        return True

    def add_ignore_word(self, word: str) -> None:
        self._ignore_words.add(self._normalize_text(word))
        logger.debug("Added '%s' to ignore list", word)

    def remove_ignore_word(self, word: str) -> None:
        self._ignore_words.discard(self._normalize_text(word))
        logger.debug("Removed '%s' from ignore list", word)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info(
            "InterruptionFilter %s",
            "enabled" if enabled else "disabled",
        )

    def __repr__(self) -> str:
        return (
            "InterruptionFilter("
            f"enabled={self._enabled}, "
            f"ignore_words_count={len(self._ignore_words)}, "
            f"case_sensitive={self._case_sensitive})"
        )
