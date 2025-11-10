"""
Interruption Handler for LiveKit Voice Agent
Filters out filler words and low-confidence speech to prevent false interruptions.
"""

import logging
import re
from typing import List, Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InterruptionConfig:
    """Configuration for interruption detection."""
    ignored_words: Set[str]
    confidence_threshold: float = 0.5
    enable_dynamic_updates: bool = False
    
    @classmethod
    def from_word_list(cls, words: List[str], confidence_threshold: float = 0.5, 
                       enable_dynamic_updates: bool = False) -> 'InterruptionConfig':
        """Create config from a list of words to ignore."""
        # Normalize words: lowercase and strip whitespace
        normalized_words = {word.lower().strip() for word in words if word.strip()}
        return cls(
            ignored_words=normalized_words,
            confidence_threshold=confidence_threshold,
            enable_dynamic_updates=enable_dynamic_updates
        )


class InterruptionHandler:
    """
    Handles interruption detection for LiveKit voice agents.
    
    This class filters out filler words and low-confidence speech segments
    to prevent false interruptions during agent speech.
    """
    
    def __init__(self, config: InterruptionConfig):
        """
        Initialize the interruption handler.
        
        Args:
            config: Configuration for interruption detection
        """
        self.config = config
        self._ignored_words = config.ignored_words.copy()
        self._confidence_threshold = config.confidence_threshold
        self._enable_dynamic_updates = config.enable_dynamic_updates
        
        logger.info(f"InterruptionHandler initialized with {len(self._ignored_words)} ignored words")
        logger.debug(f"Ignored words: {sorted(self._ignored_words)}")
        logger.info(f"Confidence threshold: {self._confidence_threshold}")
    
    def should_ignore_speech(self, text: str, confidence: Optional[float] = None) -> bool:
        """
        Determine if speech should be ignored (not treated as interruption).
        
        Args:
            text: The transcribed text from user speech
            confidence: Optional confidence score from ASR (0.0 to 1.0)
            
        Returns:
            True if speech should be ignored, False if it's a valid interruption
        """
        # Check confidence threshold first
        if confidence is not None and confidence < self._confidence_threshold:
            logger.debug(f"Ignoring low-confidence speech: '{text}' (confidence: {confidence:.2f})")
            return True
        
        # Normalize the input text
        normalized_text = self._normalize_text(text)
        
        # Check if the entire text is empty after normalization
        if not normalized_text:
            logger.debug(f"Ignoring empty text after normalization: '{text}'")
            return True
        
        # Split into words and check if all are filler words
        words = normalized_text.split()
        
        # If all words are in the ignored list, ignore this speech
        non_filler_words = [word for word in words if word not in self._ignored_words]
        
        if not non_filler_words:
            logger.debug(f"Ignoring filler-only speech: '{text}' -> words: {words}")
            return True
        
        # This is valid speech with meaningful content
        logger.debug(f"Valid interruption detected: '{text}' -> meaningful words: {non_filler_words}")
        return False
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text (lowercase, no punctuation, trimmed)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def add_ignored_word(self, word: str) -> None:
        """
        Dynamically add a word to the ignored list.
        
        Args:
            word: Word to add to ignored list
        """
        if not self._enable_dynamic_updates:
            logger.warning("Dynamic updates are disabled. Enable them in config to add words at runtime.")
            return
        
        normalized_word = word.lower().strip()
        if normalized_word and normalized_word not in self._ignored_words:
            self._ignored_words.add(normalized_word)
            logger.info(f"Added '{normalized_word}' to ignored words list")
    
    def remove_ignored_word(self, word: str) -> None:
        """
        Dynamically remove a word from the ignored list.
        
        Args:
            word: Word to remove from ignored list
        """
        if not self._enable_dynamic_updates:
            logger.warning("Dynamic updates are disabled. Enable them in config to remove words at runtime.")
            return
        
        normalized_word = word.lower().strip()
        if normalized_word in self._ignored_words:
            self._ignored_words.remove(normalized_word)
            logger.info(f"Removed '{normalized_word}' from ignored words list")
    
    def get_ignored_words(self) -> Set[str]:
        """Get the current set of ignored words."""
        return self._ignored_words.copy()
    
    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        old_threshold = self._confidence_threshold
        self._confidence_threshold = threshold
        logger.info(f"Updated confidence threshold from {old_threshold:.2f} to {threshold:.2f}")
    
    def get_stats(self) -> dict:
        """Get current handler statistics and configuration."""
        return {
            "ignored_words_count": len(self._ignored_words),
            "confidence_threshold": self._confidence_threshold,
            "dynamic_updates_enabled": self._enable_dynamic_updates,
            "ignored_words": sorted(self._ignored_words)
        }

