import os
from dotenv import load_dotenv

load_dotenv()


class InterruptionConfig:
    """Configuration for interruption handling"""

    def __init__(self):
        # Default filler words - can be overridden by environment
        self._ignored_words = {
            'uh', 'umm', 'hmm', 'haan', 'ah', 'eh', 'er', 'um',
            'like', 'you know', 'i mean', 'sort of', 'kind of'
        }

        # Real interruption triggers
        self._interruption_triggers = {
            'wait', 'stop', 'hold on', 'pause', 'just a minute',
            'no', 'not that', 'wrong', 'correction', 'actually'
        }

        # Confidence threshold for ASR
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

        # Load from environment if available
        env_ignored = os.getenv('IGNORED_WORDS')
        if env_ignored:
            self._ignored_words.update([w.strip().lower() for w in env_ignored.split(',')])

        env_triggers = os.getenv('INTERRUPTION_TRIGGERS')
        if env_triggers:
            self._interruption_triggers.update([w.strip().lower() for w in env_triggers.split(',')])

    @property
    def ignored_words(self) -> set[str]:
        return self._ignored_words.copy()

    @property
    def interruption_triggers(self) -> set[str]:
        return self._interruption_triggers.copy()

    def update_ignored_words(self, words: list[str]):
        """Dynamically update ignored words list"""
        self._ignored_words.update([w.lower() for w in words])

    def remove_ignored_word(self, word: str):
        """Remove a word from ignored list"""
        self._ignored_words.discard(word.lower())


# Global configuration instance
config = InterruptionConfig()