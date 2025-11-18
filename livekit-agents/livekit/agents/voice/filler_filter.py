"""
Filler Filter Module for LiveKit Agents

This module provides filtering logic to prevent filler words (like "umm", "hmm", "haan")
from triggering agent interruptions during ongoing speech. This is particularly useful
for voice agents that should not be interrupted by minor speech disfluencies.

The filter:
- Checks if a transcript contains only filler words
- Considers confidence scores to filter low-confidence murmurs
- Is thread-safe for async operations
- Supports configurable filler word lists
- Provides detailed logging for debugging and evaluation

Author: Raghav (LiveKit Intern Assessment)
Date: November 18, 2025
"""

from __future__ import annotations

import asyncio
import os

from ..log import logger

# Multi-language filler word database
DEFAULT_LANGUAGE_FILLERS: dict[str, list[str]] = {
    "en": ["uh", "umm", "hmm", "er", "ah", "oh", "yeah", "yep", "okay", "ok", "mm", "mhm"],
    "hi": ["haan", "arey", "accha", "theek", "yaar", "bas", "arre", "haa"],
    "es": ["eh", "este", "pues", "bueno", "entonces", "claro"],
    "fr": ["euh", "ben", "alors", "quoi", "voilà", "bah"],
    "de": ["äh", "ähm", "also", "ja", "naja", "halt"],
    "ja": ["ええと", "あの", "ま", "えっと"],
    "zh": ["嗯", "啊", "呃", "那个"],
    "pt": ["eh", "né", "então", "tipo", "bem"],
    "it": ["eh", "ehm", "allora", "cioè", "insomma"],
    "ko": ["음", "어", "그", "저"],
}


class FillerFilter:
    """
    A filter to detect and ignore filler-only speech during agent interruption handling.

    The filter helps distinguish between meaningful user interruptions and speech fillers
    like "umm", "hmm", "haan", etc. This prevents false interruptions while the agent
    is speaking.

    Attributes:
        ignored_words: List of words to consider as fillers
        min_confidence_threshold: Minimum confidence score to consider (below this = filler)
        _lock: Async lock for thread-safe operations
        _language_mode: Whether to use multi-language support ("auto", "manual", or None)
        _language_fillers: Multi-language filler word database
        _current_language: Currently detected/selected language
    """

    def __init__(
        self,
        ignored_words: list[str] | None = None,
        min_confidence_threshold: float = 0.5,
        enable_multi_language: bool = False,
        default_language: str = "en",
    ) -> None:
        """
        Initialize the FillerFilter.

        Args:
            ignored_words: List of words to treat as fillers. If None, loads from environment
                          or uses defaults.
            min_confidence_threshold: Minimum confidence score below which transcripts are
                                     considered fillers (default: 0.5)
            enable_multi_language: Enable automatic language detection and switching
            default_language: Default language code (e.g., "en", "hi", "es")
        """
        self._min_confidence_threshold = min_confidence_threshold
        self._lock = asyncio.Lock()
        self._enable_multi_language = enable_multi_language
        self._current_language = default_language
        self._language_fillers = DEFAULT_LANGUAGE_FILLERS.copy()

        if enable_multi_language:
            # Start with the default language fillers
            self.ignored_words = self._language_fillers.get(default_language, [])
            logger.info(
                "FillerFilter initialized with multi-language support",
                extra={
                    "default_language": default_language,
                    "available_languages": list(self._language_fillers.keys()),
                },
            )
        elif ignored_words is not None:
            self.ignored_words = [w.strip().lower() for w in ignored_words]
        else:
            self.ignored_words = self._load_fillers_from_env()

        logger.info(
            "FillerFilter initialized",
            extra={
                "ignored_words": self.ignored_words,
                "min_confidence_threshold": self._min_confidence_threshold,
            },
        )

    def _load_fillers_from_env(self) -> list[str]:
        """
        Load filler words from environment variable or use defaults.

        The environment variable IGNORED_WORDS should be a comma-separated list,
        e.g., "uh,umm,hmm,haan,arey"

        Returns:
            List of normalized (lowercase, trimmed) filler words
        """
        default_fillers = "uh,umm,hmm,haan,mm,mhm,er,ah,oh,yeah,yep,okay,ok"
        env_fillers = os.getenv("IGNORED_WORDS", default_fillers)

        fillers = [w.strip().lower() for w in env_fillers.split(",") if w.strip()]

        if not fillers:
            # Fallback to hardcoded defaults if env is empty
            fillers = ["uh", "umm", "hmm", "haan", "mm", "mhm", "er", "ah"]
            logger.warning(
                "No filler words configured, using fallback defaults",
                extra={"fallback_fillers": fillers},
            )

        return fillers

    def is_filler_only(
        self,
        text: str,
        confidence: float = 1.0,
        agent_is_speaking: bool = False,
        language: str | None = None,
    ) -> bool:
        """
        Determine if a transcript contains only filler words.

        This method checks:
        1. If confidence is below threshold → treat as filler
        2. If text is empty → treat as filler
        3. If all words in the text are in the ignored_words list → filler
        4. If any word is NOT in ignored_words → NOT a filler (valid speech)

        Args:
            text: The transcript text to check
            confidence: Confidence score from STT (0.0 to 1.0)
            agent_is_speaking: Whether the agent is currently speaking (for context)
            language: Language code from STT (e.g., "en", "hi", "es") - for multi-language mode

        Returns:
            True if the transcript should be ignored as filler, False otherwise
        """
        # Multi-language support: Auto-detect and switch language if enabled
        if self._enable_multi_language and language:
            self._auto_switch_language(language)

        # Low confidence is treated as filler/murmur
        if confidence < self._min_confidence_threshold:
            logger.debug(
                "Low confidence transcript treated as filler",
                extra={
                    "text": text,
                    "confidence": confidence,
                    "threshold": self._min_confidence_threshold,
                },
            )
            return True

        # Empty or whitespace-only text is considered filler
        text_cleaned = text.strip()
        if not text_cleaned:
            return True

        # Split into words and normalize
        words = self._normalize_words(text_cleaned)

        if not words:
            return True

        # Check if ALL words are fillers
        non_filler_words = []
        for word in words:
            if word not in self.ignored_words:
                non_filler_words.append(word)

        # If there are any non-filler words, this is valid speech
        is_filler = len(non_filler_words) == 0

        if is_filler:
            logger.debug(
                "Transcript contains only filler words",
                extra={
                    "text": text,
                    "words": words,
                    "confidence": confidence,
                    "agent_speaking": agent_is_speaking,
                },
            )

        return is_filler

    def _normalize_words(self, text: str) -> list[str]:
        """
        Normalize text into individual words for comparison.

        This handles:
        - Lowercasing
        - Trimming whitespace
        - Removing punctuation
        - Splitting on whitespace

        Args:
            text: Text to normalize

        Returns:
            List of normalized words
        """
        # Simple normalization - can be enhanced with more sophisticated tokenization
        # Remove common punctuation
        text_cleaned = text.lower()
        for punct in ".,!?;:\"'":
            text_cleaned = text_cleaned.replace(punct, " ")

        # Split and filter empty strings
        words = [w.strip() for w in text_cleaned.split() if w.strip()]

        return words

    async def update_ignored_words(self, words: list[str]) -> None:
        """
        Update the list of ignored filler words (thread-safe).

        This allows runtime updates to the filler list without restarting the agent.
        Useful for dynamic configuration or multi-language support.

        Args:
            words: New list of filler words to use
        """
        async with self._lock:
            self.ignored_words = [w.strip().lower() for w in words]
            logger.info(
                "Filler words updated",
                extra={"new_ignored_words": self.ignored_words},
            )

    async def add_ignored_words(self, words: list[str]) -> None:
        """
        Add words to the ignored list (thread-safe).

        Args:
            words: Words to add to the filler list
        """
        async with self._lock:
            new_words = [w.strip().lower() for w in words if w.strip()]
            for word in new_words:
                if word not in self.ignored_words:
                    self.ignored_words.append(word)
            logger.info(
                "Added filler words",
                extra={
                    "added_words": new_words,
                    "current_ignored_words": self.ignored_words,
                },
            )

    async def remove_ignored_words(self, words: list[str]) -> None:
        """
        Remove words from the ignored list (thread-safe).

        Args:
            words: Words to remove from the filler list
        """
        async with self._lock:
            words_to_remove = [w.strip().lower() for w in words]
            self.ignored_words = [
                w for w in self.ignored_words if w not in words_to_remove
            ]
            logger.info(
                "Removed filler words",
                extra={
                    "removed_words": words_to_remove,
                    "current_ignored_words": self.ignored_words,
                },
            )

    def get_ignored_words(self) -> list[str]:
        """
        Get the current list of ignored filler words.

        Returns:
            Copy of the current ignored words list
        """
        return self.ignored_words.copy()

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the minimum confidence threshold.

        Args:
            threshold: New minimum confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")

        self._min_confidence_threshold = threshold
        logger.info(
            "Confidence threshold updated",
            extra={"new_threshold": threshold},
        )

    # ========== BONUS FEATURE 1: Dynamic Updates ==========

    async def update_fillers_dynamic(self, add: list[str] | None = None, remove: list[str] | None = None) -> dict[str, any]:
        """
        Dynamically update filler words at runtime (REST API compatible).

        This method allows runtime updates via REST/WebSocket endpoints:
        POST /update_filler
        body: { "add": ["arey"], "remove": ["haan"] }

        Args:
            add: List of words to add to the filler list
            remove: List of words to remove from the filler list

        Returns:
            Dictionary with status and updated filler list
        """
        async with self._lock:
            result = {
                "status": "success",
                "added": [],
                "removed": [],
                "current_fillers": []
            }

            if add:
                new_words = [w.strip().lower() for w in add if w.strip()]
                for word in new_words:
                    if word not in self.ignored_words:
                        self.ignored_words.append(word)
                        result["added"].append(word)

                logger.info(
                    "[DYNAMIC_UPDATE] Added filler words",
                    extra={"added_words": result["added"]},
                )

            if remove:
                words_to_remove = [w.strip().lower() for w in remove]
                original_count = len(self.ignored_words)
                self.ignored_words = [
                    w for w in self.ignored_words if w not in words_to_remove
                ]
                removed_count = original_count - len(self.ignored_words)
                result["removed"] = [w for w in words_to_remove if w in result["removed"] or removed_count > 0]

                logger.info(
                    "[DYNAMIC_UPDATE] Removed filler words",
                    extra={"removed_words": words_to_remove},
                )

            result["current_fillers"] = self.ignored_words.copy()

            logger.info(
                "[DYNAMIC_UPDATE] Filler list updated",
                extra={
                    "total_fillers": len(self.ignored_words),
                    "operation": f"added={len(result['added'])}, removed={len(result['removed'])}",
                },
            )

            return result

    # ========== BONUS FEATURE 2: Multi-Language Support ==========

    def _auto_switch_language(self, language: str) -> None:
        """
        Automatically switch filler words based on detected language.

        Args:
            language: Language code from STT (e.g., "en", "hi", "es")
        """
        if not self._enable_multi_language:
            return

        # Normalize language code (e.g., "en-US" -> "en")
        lang_code = language.split("-")[0].lower() if language else self._current_language

        # Only switch if language changed
        if lang_code == self._current_language:
            return

        # Check if we have fillers for this language
        if lang_code in self._language_fillers:
            old_language = self._current_language
            self._current_language = lang_code
            self.ignored_words = self._language_fillers[lang_code].copy()

            logger.info(
                "[MULTI_LANG] Language switched",
                extra={
                    "from": old_language,
                    "to": lang_code,
                    "new_fillers": self.ignored_words,
                },
            )
        else:
            logger.debug(
                "[MULTI_LANG] Unsupported language, using current fillers",
                extra={
                    "detected_language": lang_code,
                    "current_language": self._current_language,
                    "available_languages": list(self._language_fillers.keys()),
                },
            )

    def add_language_fillers(self, language: str, fillers: list[str]) -> None:
        """
        Add or update filler words for a specific language.

        Args:
            language: Language code (e.g., "en", "hi", "es")
            fillers: List of filler words for this language
        """
        normalized_fillers = [w.strip().lower() for w in fillers if w.strip()]
        self._language_fillers[language] = normalized_fillers

        logger.info(
            "[MULTI_LANG] Added/updated language fillers",
            extra={
                "language": language,
                "fillers": normalized_fillers,
            },
        )

        # If this is the current language, update active fillers
        if self._enable_multi_language and language == self._current_language:
            self.ignored_words = normalized_fillers.copy()

    def switch_language(self, language: str) -> bool:
        """
        Manually switch to a different language.

        Args:
            language: Language code to switch to

        Returns:
            True if switch was successful, False if language not available
        """
        if language in self._language_fillers:
            self._current_language = language
            self.ignored_words = self._language_fillers[language].copy()

            logger.info(
                "[MULTI_LANG] Manually switched language",
                extra={
                    "language": language,
                    "fillers": self.ignored_words,
                },
            )
            return True

        logger.warning(
            "[MULTI_LANG] Cannot switch to unsupported language",
            extra={
                "requested_language": language,
                "available_languages": list(self._language_fillers.keys()),
            },
        )
        return False

    def get_available_languages(self) -> list[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        return list(self._language_fillers.keys())

    def get_current_language(self) -> str:
        """
        Get the currently active language.

        Returns:
            Current language code
        """
        return self._current_language

    def get_language_fillers(self, language: str) -> list[str] | None:
        """
        Get filler words for a specific language.

        Args:
            language: Language code

        Returns:
            List of filler words or None if language not supported
        """
        return self._language_fillers.get(language)
