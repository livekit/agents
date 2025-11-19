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

# Multi-language filler word database with single words and phrases
DEFAULT_LANGUAGE_FILLERS: dict[str, dict[str, list[str]]] = {
    "en": {
        "single": ["uh", "umm", "hmm", "er", "ah", "oh", "yeah", "yep", "okay", "ok", 
                   "mm", "mhm", "well", "like", "so", "right", "um", "huh"],
        "phrases": ["you know", "i mean", "you see", "kind of", "sort of", 
                    "uh huh", "mm hmm", "you know what i mean"]
    },
    "hi": {
        "single": ["haan", "arey", "accha", "theek", "yaar", "bas", "arre", "haa", "ji", "hai"],
        "phrases": ["theek hai", "haan ji", "accha theek hai", "bas yaar"]
    },
    "es": {
        "single": ["eh", "este", "pues", "bueno", "entonces", "claro"],
        "phrases": ["o sea", "pues si"]
    },
    "fr": {
        "single": ["euh", "ben", "alors", "quoi", "voilà", "bah"],
        "phrases": ["tu vois", "tu sais"]
    },
    "de": {
        "single": ["äh", "ähm", "also", "ja", "naja", "halt"],
        "phrases": ["weißt du"]
    },
    "ja": {
        "single": ["ええと", "あの", "ま", "えっと"],
        "phrases": []
    },
    "zh": {
        "single": ["嗯", "啊", "呃", "那个"],
        "phrases": []
    },
    "pt": {
        "single": ["eh", "né", "então", "tipo", "bem"],
        "phrases": ["sabe"]
    },
    "it": {
        "single": ["eh", "ehm", "allora", "cioè", "insomma"],
        "phrases": ["cioè"]
    },
    "ko": {
        "single": ["음", "어", "그", "저"],
        "phrases": []
    },
}


class FillerFilter:
    """
    A filter to detect and ignore filler-only speech during agent interruption handling.

    The filter helps distinguish between meaningful user interruptions and speech fillers
    like "umm", "hmm", "haan", etc. This prevents false interruptions while the agent
    is speaking.

    Attributes:
        ignored_words: List of single words to consider as fillers
        ignored_phrases: List of multi-word phrases to consider as fillers
        min_confidence_threshold: Minimum confidence score to consider (below this = filler)
        _lock: Async lock for thread-safe operations
        _enable_multi_language: Whether to use multi-language support
        _language_fillers: Multi-language filler word database
        _current_language: Currently detected/selected language
        _context_aware: Enable context-aware classification
    """

    def __init__(
        self,
        ignored_words: list[str] | None = None,
        min_confidence_threshold: float = 0.5,
        enable_multi_language: bool = False,
        default_language: str = "en",
        context_aware: bool = True,
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
            context_aware: Enable context-aware classification for ambiguous words
        """
        self._min_confidence_threshold = min_confidence_threshold
        self._lock = asyncio.Lock()
        self._enable_multi_language = enable_multi_language
        self._current_language = default_language
        self._language_fillers = DEFAULT_LANGUAGE_FILLERS.copy()
        self._context_aware = context_aware

        if enable_multi_language:
            # Start with the default language fillers
            lang_data = self._language_fillers.get(default_language, {"single": [], "phrases": []})
            self.ignored_words = lang_data.get("single", [])
            self.ignored_phrases = lang_data.get("phrases", [])
            logger.info(
                "FillerFilter initialized with multi-language support",
                extra={
                    "default_language": default_language,
                    "available_languages": list(self._language_fillers.keys()),
                    "context_aware": context_aware,
                },
            )
        elif ignored_words is not None:
            self.ignored_words = [w.strip().lower() for w in ignored_words]
            self.ignored_phrases = []
        else:
            fillers_from_env = self._load_fillers_from_env()
            # Load default English fillers with phrases
            if not fillers_from_env:
                lang_data = self._language_fillers.get("en", {"single": [], "phrases": []})
                self.ignored_words = lang_data.get("single", [])
                self.ignored_phrases = lang_data.get("phrases", [])
            else:
                self.ignored_words = fillers_from_env
                self.ignored_phrases = []

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

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        This handles:
        - Lowercasing
        - Hyphen to space conversion (uh-huh -> uh huh)
        - Removing punctuation
        - Whitespace normalization

        Args:
            text: Text to normalize

        Returns:
            Normalized text string
        """
        text_lower = text.lower().strip()
        
        # Convert hyphens to spaces for hyphenated fillers
        text_lower = text_lower.replace('-', ' ')
        
        # Remove punctuation
        punctuation = ".,!?;:\"'…"
        for punct in punctuation:
            text_lower = text_lower.replace(punct, ' ')
        
        # Normalize whitespace
        text_lower = ' '.join(text_lower.split())
        
        return text_lower

    def _extract_words(self, text: str) -> list[str]:
        """
        Extract individual words from normalized text.

        Args:
            text: Normalized text

        Returns:
            List of words
        """
        return [w.strip() for w in text.split() if w.strip()]

    def _check_phrase_fillers(self, text: str) -> bool | None:
        """
        Check for multi-word filler phrases.

        Args:
            text: Normalized text

        Returns:
            True if text is filler-only phrase
            False if text contains non-filler content with phrases
            None if no phrase match (continue with word-level analysis)
        """
        if not self.ignored_phrases:
            return None

        # Exact phrase match
        if text in self.ignored_phrases:
            logger.debug("Exact phrase match", extra={"text": text})
            return True

        # Check for phrase combinations
        remaining_text = text
        matched_phrases = []

        for phrase in sorted(self.ignored_phrases, key=len, reverse=True):
            if phrase in remaining_text:
                matched_phrases.append(phrase)
                remaining_text = remaining_text.replace(phrase, ' ', 1)

        if matched_phrases:
            remaining_words = self._extract_words(remaining_text)
            if all(w in self.ignored_words for w in remaining_words):
                logger.debug(
                    "Phrase fillers detected",
                    extra={"phrases": matched_phrases, "text": text}
                )
                return True
            else:
                # Has meaningful content along with phrases
                return False

        return None

    def _context_aware_classification(self, text: str, words: list[str]) -> bool:
        """
        Context-aware classification for ambiguous words.

        Some words like "well", "like", "okay", "so" can be fillers when standalone
        but meaningful when part of a sentence.

        Args:
            text: Full normalized text
            words: List of words in text

        Returns:
            True if text is filler-only, False otherwise
        """
        # Single word case
        if len(words) == 1:
            word = words[0]
            # Ambiguous words that are fillers when standalone
            ambiguous_fillers = {"well", "like", "okay", "so", "right"}
            if word in ambiguous_fillers:
                return True
            return word in self.ignored_words

        # Multiple words - check if ALL are fillers
        filler_count = sum(1 for w in words if w in self.ignored_words)
        non_filler_count = len(words) - filler_count

        # All fillers
        if non_filler_count == 0:
            logger.debug(
                "All words are fillers",
                extra={"text": text, "words": words}
            )
            return True

        # Mixed content - has meaningful words
        return False

    def _normalize_words(self, text: str) -> list[str]:
        """
        Legacy method for backward compatibility.
        Normalizes text and returns words.

        Args:
            text: Text to normalize

        Returns:
            List of normalized words
        """
        normalized = self._normalize_text(text)
        return self._extract_words(normalized)

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
            lang_data = self._language_fillers[lang_code]
            self.ignored_words = lang_data.get("single", []).copy()
            self.ignored_phrases = lang_data.get("phrases", []).copy()

            logger.info(
                "[MULTI_LANG] Language switched",
                extra={
                    "from": old_language,
                    "to": lang_code,
                    "words": len(self.ignored_words),
                    "phrases": len(self.ignored_phrases),
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
            lang_data = self._language_fillers[language]
            self.ignored_words = lang_data.get("single", []).copy()
            self.ignored_phrases = lang_data.get("phrases", []).copy()

            logger.info(
                "[MULTI_LANG] Manually switched language",
                extra={
                    "language": language,
                    "words": len(self.ignored_words),
                    "phrases": len(self.ignored_phrases),
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
