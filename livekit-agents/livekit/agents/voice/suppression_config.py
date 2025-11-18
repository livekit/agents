"""Filler Word Suppression Configuration.

This module provides configuration for suppressing filler words and speech
disfluencies in voice agent interactions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal


# Common acknowledgment words that should NOT be suppressed when used alone
# These are meaningful responses, not just fillers
ACKNOWLEDGMENT_WORDS = {
    "okay", "ok", "yeah", "yes", "yep", "sure", "right", "correct",
    "alright", "fine", "good", "great", "perfect", "excellent",
    "no", "nope", "nah", "never", "absolutely", "definitely"
}


# Default suppression patterns for multiple languages (regex-based approach)
DEFAULT_SUPPRESSION_PATTERNS = {
    "en": r"\b(uh+|um+|hmm+|er+|ah+|oh+|yeah|yep|okay|ok|mm+|mhm+)\b",
    "hi": r"\b(haan|arey|yaar|bas|theek|accha|अच्छा|हाँ|हां|अरे|यार|बस|ठीक)\b",
    "es": r"\b(eh+|em+|este|pues|bueno|claro|sí|vale)\b",
    "fr": r"\b(euh+|ben|alors|bon|hein|voilà|quoi)\b",
    "de": r"\b(äh+|ähm+|also|halt|eben|ja|genau)\b",
    "ja": r"\b(ano+|eto+|ma+|ne+|sa+|un+|ああ|えと|まあ|ね|さあ)\b",
    "zh": r"\b(en+|uh+|嗯|啊|哦|那个|这个|就是)\b",
    "pt": r"\b(eh+|né|então|tipo|assim|sabe|pois)\b",
    "it": r"\b(ehm+|allora|cioè|quindi|insomma|ecco|sì)\b",
    "ko": r"\b(uh+|um+|그|저|음|어|아|네|예)\b",
}


class SuppressionConfig:
    """Configuration class for filler word suppression.

    Manages suppression word lists, confidence thresholds,
    multi-language regex patterns, and dynamic configuration loading.
    """
    
    def __init__(
        self,
        suppression_words: set[str] | None = None,
        min_confidence: float = 0.5,
        enable_patterns: bool = True,
        detection_mode: Literal["strict", "balanced", "lenient"] = "balanced",
        min_filler_ratio: float = 0.7,
    ):
        """Initialize suppression configuration.

        Args:
            suppression_words: Set of words to suppress (case-insensitive)
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            enable_patterns: Whether to use regex pattern matching
            detection_mode: Detection sensitivity (strict/balanced/lenient)
            min_filler_ratio: Minimum ratio of filler words to total words (0.0-1.0)
        """
        self._suppression_words: set[str] = suppression_words or set()
        self._min_confidence = min_confidence
        self._enable_patterns = enable_patterns
        self._detection_mode = detection_mode
        self._min_filler_ratio = min_filler_ratio

        # Compile regex patterns for efficiency
        self._compiled_patterns: dict[str, re.Pattern] = {}
        if self._enable_patterns:
            self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for all supported languages."""
        for lang, pattern in DEFAULT_SUPPRESSION_PATTERNS.items():
            try:
                self._compiled_patterns[lang] = re.compile(
                    pattern,
                    re.IGNORECASE | re.UNICODE
                )
            except re.error as e:
                print(f"[SUPPRESSION] Error compiling pattern for {lang}: {e}")
    
    def is_suppressed_word(self, word: str) -> bool:
        """Check if a single word should be suppressed.
        
        Args:
            word: The word to check (case-insensitive)
            
        Returns:
            True if word is a filler word
        """
        word_lower = word.lower().strip()
        
        # Check custom suppression words
        if word_lower in self._suppression_words:
            return True
        
        # Check against all language patterns for single words
        if self._enable_patterns and self._compiled_patterns:
            for pattern in self._compiled_patterns.values():
                # Match full word only using word boundaries
                if pattern.fullmatch(word_lower):
                    return True
        
        return False
    
    def matches_pattern(self, text: str, language: str | None = None) -> bool:
        """Check if text matches a filler pattern for the given language.

        Args:
            text: Text to check
            language: ISO language code (e.g., 'en', 'hi', 'es')

        Returns:
            True if text matches a filler pattern
        """
        if not self._enable_patterns or not language:
            return False

        # Get compiled pattern for language
        pattern = self._compiled_patterns.get(language)
        if not pattern:
            return False

        # Check if entire text matches the pattern
        match = pattern.search(text.lower())
        return match is not None

    def calculate_filler_ratio(self, text: str) -> float:
        """Calculate the ratio of filler words to total words.

        Args:
            text: Text to analyze

        Returns:
            Ratio between 0.0 and 1.0
        """
        words = text.strip().lower().split()
        if not words:
            return 0.0

        filler_count = sum(1 for word in words if self.is_suppressed_word(word))
        return filler_count / len(words)

    def is_only_filler_words(self, text: str) -> bool:
        """Check if text contains only filler words using strict matching.

        Args:
            text: Text to check

        Returns:
            True if all words are fillers
        """
        words = text.strip().lower().split()
        if not words:
            return True

        return all(self.is_suppressed_word(word) for word in words)

    def should_suppress_advanced(
        self,
        text: str,
        confidence: float,
        language: str | None = None,
    ) -> tuple[bool, str]:
        """Advanced suppression logic using multiple algorithms.

        Combines multiple detection strategies based on detection mode.

        Args:
            text: Transcript text
            confidence: Confidence score from STT
            language: Language code for pattern matching

        Returns:
            Tuple of (should_suppress: bool, reason: str)
        """
        # Algorithm 1: Confidence thresholding
        if confidence < self._min_confidence:
            return True, f"Low confidence ({confidence:.2f} < {self._min_confidence})"

        # Algorithm 2: Empty text detection
        cleaned_text = text.strip()
        if not cleaned_text:
            return True, "Empty text"
        
        # Algorithm 6: Single acknowledgment word detection
        # If user says just "okay", "yeah", etc., it's likely intentional
        words = cleaned_text.lower().split()
        if len(words) == 1 and words[0] in ACKNOWLEDGMENT_WORDS:
            return False, f"Single acknowledgment word '{words[0]}' - allowing"

        # Algorithm 3: Pattern matching for pure filler detection
        if language and self._enable_patterns:
            if self.matches_pattern(cleaned_text, language):
                # Pattern matched - now check if it's ALL fillers
                if self.is_only_filler_words(cleaned_text):
                    return True, "Pattern match - only filler words"

        # Algorithm 4: Filler ratio analysis
        filler_ratio = self.calculate_filler_ratio(cleaned_text)
        
        # Check for acknowledgment + filler combinations (e.g., "hmm okay")
        # If any word is an acknowledgment, treat it as meaningful content
        has_acknowledgment = any(word in ACKNOWLEDGMENT_WORDS for word in words)
        if has_acknowledgment and len(words) <= 3:
            # Short phrase with acknowledgment - likely meaningful
            return False, f"Contains acknowledgment word - allowing"

        # Apply mode-specific thresholds
        if self._detection_mode == "strict":
            # Strict: Only suppress if 100% fillers
            if filler_ratio >= 1.0:
                return True, f"Strict mode - all words are fillers (ratio: {filler_ratio:.2f})"

        elif self._detection_mode == "balanced":
            # Balanced: Suppress if >= configured ratio (default 70%)
            if filler_ratio >= self._min_filler_ratio:
                return True, f"Balanced mode - high filler ratio ({filler_ratio:.2f} >= {self._min_filler_ratio})"

        elif self._detection_mode == "lenient":
            # Lenient: Suppress if > 80% fillers
            if filler_ratio >= 0.8:
                return True, f"Lenient mode - very high filler ratio ({filler_ratio:.2f})"

        # Algorithm 5: Short utterance analysis
        words = cleaned_text.split()
        if len(words) <= 2:
            # For very short utterances, be more strict
            if all(self.is_suppressed_word(w) for w in words):
                return True, f"Short utterance - all {len(words)} words are fillers"

        # Not suppressed
        non_filler_words = [w for w in words if not self.is_suppressed_word(w)]
        return False, f"Contains {len(non_filler_words)} non-filler words (ratio: {filler_ratio:.2f})"

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self._min_confidence
    
    def set_min_confidence(self, threshold: float) -> None:
        """
        Update minimum confidence threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._min_confidence = threshold
    
    def get_suppression_words(self) -> set[str]:
        """Get current suppression words."""
        return self._suppression_words.copy()
    
    def add_suppression_words(self, words: list[str]) -> None:
        """
        Add words to suppression list.
        
        Args:
            words: List of words to add
        """
        self._suppression_words.update(w.lower() for w in words)
    
    def remove_suppression_words(self, words: list[str]) -> None:
        """
        Remove words from suppression list.
        
        Args:
            words: List of words to remove
        """
        self._suppression_words.difference_update(w.lower() for w in words)
    
    def clear_suppression_words(self) -> None:
        """Clear all suppression words."""
        self._suppression_words.clear()
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load configuration from JSON file.
        
        Expected JSON structure:
        {
            "suppression_words": ["uh", "umm", "hmm"],
            "min_confidence": 0.5,
            "enable_patterns": true
        }
        
        Args:
            filepath: Path to JSON configuration file
        """
        try:
            path = Path(filepath)
            if not path.exists():
                print(f"[SUPPRESSION] Config file not found: {filepath}")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load suppression words
            if "suppression_words" in config:
                words = config["suppression_words"]
                if isinstance(words, list):
                    self._suppression_words = set(w.lower() for w in words)
            
            # Load confidence threshold
            if "min_confidence" in config:
                self._min_confidence = float(config["min_confidence"])
            
            # Load pattern matching flag
            if "enable_patterns" in config:
                self._enable_patterns = bool(config["enable_patterns"])
                if self._enable_patterns and not self._compiled_patterns:
                    self._compile_patterns()

            # Load detection mode
            if "detection_mode" in config:
                mode = config["detection_mode"]
                if mode in ("strict", "balanced", "lenient"):
                    self._detection_mode = mode

            # Load filler ratio threshold
            if "min_filler_ratio" in config:
                self._min_filler_ratio = float(config["min_filler_ratio"])

            print(f"[SUPPRESSION] Loaded config from {filepath}")
            print(f"[SUPPRESSION] Words: {len(self._suppression_words)}, "
                  f"Confidence: {self._min_confidence}, Mode: {self._detection_mode}")

        except Exception as e:
            print(f"[SUPPRESSION] Error loading config: {e}")
    
    def load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Supported variables:
        - SUPPRESSION_WORDS: Comma-separated list
        - SUPPRESSION_MIN_CONFIDENCE: Float value
        - SUPPRESSION_ENABLE_PATTERNS: Boolean
        """
        # Load words from environment
        words_env = os.getenv("SUPPRESSION_WORDS")
        if words_env:
            words = [w.strip() for w in words_env.split(",")]
            self._suppression_words = set(w.lower() for w in words if w)
        
        # Load confidence
        confidence_env = os.getenv("SUPPRESSION_MIN_CONFIDENCE")
        if confidence_env:
            try:
                self._min_confidence = float(confidence_env)
            except ValueError:
                print(f"[SUPPRESSION] Invalid confidence value: {confidence_env}")
        
        # Load pattern flag
        patterns_env = os.getenv("SUPPRESSION_ENABLE_PATTERNS")
        if patterns_env:
            self._enable_patterns = patterns_env.lower() in ("true", "1", "yes")
    
    def reload(self) -> None:
        """
        Reload configuration.
        
        This can be called by a file watcher when config changes.
        """
        print("[SUPPRESSION] Reloading configuration...")
        # Re-compile patterns if needed
        if self._enable_patterns and not self._compiled_patterns:
            self._compile_patterns()
