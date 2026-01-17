"""
Intelligent Interruption Filter for LiveKit Voice Agent

This module implements context-aware interruption filtering to distinguish between
passive acknowledgements (backchannel) and active interruptions based on agent state.
"""

import logging
import re
from typing import Set

logger = logging.getLogger(__name__)


class InterruptionFilter:
    """
    Filters interruptions based on transcript content and agent speaking state.
    
    Prevents the agent from stopping mid-speech when the user provides passive
    feedback (backchannel words like "yeah", "ok", "hmm"), while still allowing
    legitimate interruptions.
    """
    
    # Backchannel words that should be ignored when agent is speaking
    DEFAULT_BACKCHANNEL_WORDS: Set[str] = {
        "yeah", "yea", "yes", "yep", "yup",
        "ok", "okay", "alright", "aight",
        "hmm", "hm", "mhm", "mmhmm", "uh-huh", "uhuh",
        "right", "sure", "gotcha", "got it",
        "aha", "ah", "oh", "ooh",
        "mm", "mhmm", "huh"
    }
    
    # Command words that should always trigger interruption
    DEFAULT_COMMAND_WORDS: Set[str] = {
        "stop", "wait", "hold", "pause",
        "no", "nope", "don't",
        "hold on", "wait a second", "wait a minute",
        "hang on", "one second", "one minute"
    }
    
    def __init__(
        self,
        backchannel_words: Set[str] | None = None,
        command_words: Set[str] | None = None,
        min_word_threshold: int = 5
    ):
        """
        Initialize the interruption filter.
        
        Args:
            backchannel_words: Custom set of backchannel words (overrides defaults)
            command_words: Custom set of command words (overrides defaults)
            min_word_threshold: If transcript has more than this many words and
                              isn't purely backchannel, allow interruption
        """
        self.backchannel_words = backchannel_words or self.DEFAULT_BACKCHANNEL_WORDS
        self.command_words = command_words or self.DEFAULT_COMMAND_WORDS
        self.min_word_threshold = min_word_threshold
        
        logger.info(
            f"InterruptionFilter initialized with {len(self.backchannel_words)} "
            f"backchannel words and {len(self.command_words)} command words"
        )
    
    def should_interrupt(self, transcript: str, agent_is_speaking: bool) -> bool:
        """
        Determine if the user input should interrupt the agent.
        
        Args:
            transcript: The user's transcribed speech
            agent_is_speaking: True if agent is currently speaking, False otherwise
        
        Returns:
            True if the agent should be interrupted, False if input should be ignored
        """
        if not transcript or not transcript.strip():
            # Empty transcript - no interruption
            return False
        
        # Normalize the transcript
        normalized = self._normalize_text(transcript)
        
        # Extract words
        words = self._extract_words(normalized)
        
        if not words:
            return False
        
        # Check for explicit command words (always interrupt)
        if self._contains_command(normalized, words):
            logger.debug(
                f"Interruption ALLOWED: command word detected - "
                f"transcript='{transcript}', agent_speaking={agent_is_speaking}"
            )
            return True
        
        # If agent is not speaking, always process the input
        if not agent_is_speaking:
            logger.debug(
                f"Interruption ALLOWED: agent not speaking - transcript='{transcript}'"
            )
            return True
        
        # Agent IS speaking - check if this is pure backchannel
        if self._is_pure_backchannel(words):
            logger.info(
                f"Interruption BLOCKED: pure backchannel detected - "
                f"transcript='{transcript}', agent_speaking={agent_is_speaking}"
            )
            return False
        
        # Mixed or substantial input while agent is speaking - allow interruption
        logger.debug(
            f"Interruption ALLOWED: substantial input - "
            f"transcript='{transcript}', agent_speaking={agent_is_speaking}"
        )
        return True
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing and removing extra whitespace."""
        return " ".join(text.lower().strip().split())
    
    def _extract_words(self, text: str) -> list[str]:
        """
        Extract words from text, removing punctuation.
        
        Returns a list of cleaned words.
        """
        # Remove punctuation but keep hyphens in compound words
        text = re.sub(r'[^\w\s-]', ' ', text)
        # Split and filter empty strings
        words = [w.strip() for w in text.split() if w.strip()]
        return words
    
    def _contains_command(self, normalized_text: str, words: list[str]) -> bool:
        """
        Check if the text contains any command words.
        
        Checks both individual words and multi-word phrases.
        """
        # Check multi-word command phrases first
        for command in self.command_words:
            if ' ' in command and command in normalized_text:
                return True
        
        # Check individual command words
        for word in words:
            if word in self.command_words:
                return True
        
        return False
    
    def _is_pure_backchannel(self, words: list[str]) -> bool:
        """
        Check if all words in the list are backchannel words.
        
        Args:
            words: List of cleaned words from user input
        
        Returns:
            True if ALL words are backchannel, False otherwise
        """
        if not words:
            return False
        
        # If there are too many words, it's probably not pure backchannel
        if len(words) > self.min_word_threshold:
            return False
        
        # Check if all words are in the backchannel set
        for word in words:
            # First check if the whole word (including hyphens) is in the set
            if word in self.backchannel_words:
                continue
            # If not, try splitting hyphenated words and check each part
            elif '-' in word:
                parts = word.split('-')
                if not all(part in self.backchannel_words for part in parts if part):
                    return False
            else:
                # Word is not backchannel
                return False
        return True
    
    def is_backchannel_word(self, word: str) -> bool:
        """Check if a single word is a backchannel word."""
        normalized = word.lower().strip()
        # First check if the whole word is in the set
        if normalized in self.backchannel_words:
            return True
        # If not, try splitting hyphenated words
        if '-' in normalized:
            parts = normalized.split('-')
            return all(part in self.backchannel_words for part in parts if part)
        return False
    
    def add_backchannel_word(self, word: str) -> None:
        """Add a custom backchannel word to the filter."""
        self.backchannel_words.add(word.lower().strip())
        logger.debug(f"Added backchannel word: '{word}'")
    
    def add_command_word(self, word: str) -> None:
        """Add a custom command word to the filter."""
        self.command_words.add(word.lower().strip())
        logger.debug(f"Added command word: '{word}'")
    
    def remove_backchannel_word(self, word: str) -> None:
        """Remove a backchannel word from the filter."""
        self.backchannel_words.discard(word.lower().strip())
        logger.debug(f"Removed backchannel word: '{word}'")
    
    def remove_command_word(self, word: str) -> None:
        """Remove a command word from the filter."""
        self.command_words.discard(word.lower().strip())
        logger.debug(f"Removed command word: '{word}'")
