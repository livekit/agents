"""
Configuration manager for interruption filtering.
Handles ignored words list and confidence thresholds.
"""
import os
import json
from typing import List, Set
from threading import Lock


class ConfigManager:
    """Manages configuration for interruption filtering."""
    
    def __init__(self):
        self._lock = Lock()
        self._ignored_words = self._load_ignored_words()
        self._confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
        self._enable_dynamic_updates = os.getenv('ENABLE_DYNAMIC_UPDATES', 'false').lower() == 'true'
        
    def _load_ignored_words(self) -> Set[str]:
        """Load ignored words from environment or use defaults."""
        ignored_words_env = os.getenv('IGNORED_WORDS', '')
        
        if ignored_words_env:
            try:
                words = json.loads(ignored_words_env)
                return set(word.lower().strip() for word in words)
            except json.JSONDecodeError:
                words = ignored_words_env.split(',')
                return set(word.lower().strip() for word in words)
        
        # Default filler words (English + Hindi)
        return {
            'uh', 'uhh', 'um', 'umm', 'hmm', 'hmmm', 'mhm',
            'ah', 'ahh', 'oh', 'haan', 'haan-haan', 'haan haan',
            'er', 'err', 'erm', 'like', 'you know'
        }
    
    def is_ignored_word(self, text: str) -> bool:
        """Check if the text should be ignored."""
        with self._lock:
            normalized_text = text.lower().strip()
            return normalized_text in self._ignored_words
    
    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for filtering."""
        return self._confidence_threshold
    
    def update_ignored_words(self, words: List[str]) -> None:
        """Dynamically update the ignored words list."""
        if not self._enable_dynamic_updates:
            raise RuntimeError("Dynamic updates are not enabled")
        
        with self._lock:
            self._ignored_words = set(word.lower().strip() for word in words)
    
    def add_ignored_word(self, word: str) -> None:
        """Add a word to the ignored list."""
        if not self._enable_dynamic_updates:
            raise RuntimeError("Dynamic updates are not enabled")
        
        with self._lock:
            self._ignored_words.add(word.lower().strip())
    
    def remove_ignored_word(self, word: str) -> None:
        """Remove a word from the ignored list."""
        if not self._enable_dynamic_updates:
            raise RuntimeError("Dynamic updates are not enabled")
        
        with self._lock:
            self._ignored_words.discard(word.lower().strip())
    
    def get_ignored_words(self) -> List[str]:
        """Get a copy of the ignored words list."""
        with self._lock:
            return list(self._ignored_words)
