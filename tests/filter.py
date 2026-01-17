from __future__ import annotations

import logging
import os

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
        self._enabled = enabled
        self._case_sensitive = case_sensitive

        # Load ignore words from parameter, environment variable, or default
        if ignore_words is not None:
            self._ignore_words = set(ignore_words)
        else:
            env_words = os.getenv("LIVEKIT_INTERRUPTION_IGNORE_WORDS")
            if env_words:
                self._ignore_words = {w.strip() for w in env_words.split(",")}
                logger.info(
                    "Loaded %d ignore words from environment variable",
                    len(self._ignore_words),
                )
            else:
                self._ignore_words = self.DEFAULT_IGNORE_WORDS.copy()

        if not self._case_sensitive:
            self._ignore_words = {word.lower() for word in self._ignore_words}

        logger.info(
            "InterruptionFilter initialized with %d ignore words, enabled=%s, case_sensitive=%s",
            len(self._ignore_words),
            self._enabled,
            self._case_sensitive,
        )

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

        normalized_text = text.strip()
        if not self._case_sensitive:
            normalized_text = normalized_text.lower()

        normalized_text = (
            normalized_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        )

        words = normalized_text.split()
        if not words:
            return False

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
        normalized_word = word if self._case_sensitive else word.lower()
        self._ignore_words.add(normalized_word)
        logger.debug("Added '%s' to ignore list", word)

    def remove_ignore_word(self, word: str) -> None:
        normalized_word = word if self._case_sensitive else word.lower()
        self._ignore_words.discard(normalized_word)
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
