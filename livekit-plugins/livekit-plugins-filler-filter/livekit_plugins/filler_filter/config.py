"""
Configuration module for Filler Filter
Manages filler words list and filtering parameters with runtime update support
"""
import os
import json
import asyncio
from typing import List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class FillerFilterConfig:
    """Configuration for filler word filtering"""
    
    # Default filler words (English + Hindi mix)
    ignored_words: Set[str] = field(default_factory=lambda: {
        'uh', 'um', 'umm', 'ummm', 'hmm', 'hmmm', 'haan', 'haan ji',
        'err', 'ah', 'oh', 'eh', 'mhm', 'mm', 'mmm',
        'yeah yeah', 'like', 'you know'
    })
    
    # Confidence threshold for filtering (0.0 - 1.0)
    confidence_threshold: float = 0.3
    
    # Enable/disable filtering
    enabled: bool = True
    
    # Debug logging
    debug_mode: bool = False
    
    # Lock for thread-safe updates
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize from environment variables if available"""
        # Load from environment variable FILLER_WORDS (JSON array or comma-separated)
        env_words = os.getenv('FILLER_WORDS')
        if env_words:
            try:
                # Try parsing as JSON array
                words = json.loads(env_words)
                if isinstance(words, list):
                    self.ignored_words = set(w.lower().strip() for w in words)
            except json.JSONDecodeError:
                # Fall back to comma-separated
                self.ignored_words = set(w.lower().strip() for w in env_words.split(','))
        
        # Load confidence threshold
        threshold = os.getenv('FILLER_CONFIDENCE_THRESHOLD')
        if threshold:
            try:
                self.confidence_threshold = float(threshold)
            except ValueError:
                pass
        
        # Load debug mode
        self.debug_mode = os.getenv('FILLER_DEBUG', 'false').lower() == 'true'
    
    async def add_words(self, words: List[str]):
        """Add words to the ignored list (thread-safe)"""
        async with self._lock:
            self.ignored_words.update(w.lower().strip() for w in words)
    
    async def remove_words(self, words: List[str]):
        """Remove words from the ignored list (thread-safe)"""
        async with self._lock:
            self.ignored_words.difference_update(w.lower().strip() for w in words)
    
    async def update_words(self, words: List[str]):
        """Replace the entire ignored words list (thread-safe)"""
        async with self._lock:
            self.ignored_words = set(w.lower().strip() for w in words)
    
    def is_ignored(self, word: str) -> bool:
        """Check if a word should be ignored"""
        return word.lower().strip() in self.ignored_words


# Global singleton instance
_config_instance: Optional[FillerFilterConfig] = None


def get_config() -> FillerFilterConfig:
    """Get the global configuration instance (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = FillerFilterConfig()
    return _config_instance


def reset_config():
    """Reset configuration to defaults (useful for testing)"""
    global _config_instance
    _config_instance = FillerFilterConfig()
    return _config_instance
