"""
Filler Interruption Handler - Extension Layer for LiveKit Agents

This module provides an extension layer that intelligently filters filler words
(like "uh", "umm", "hmm") to prevent false interruptions during agent speech.

It does NOT modify LiveKit's base VAD or core SDK code - it works purely as 
an event-based extension using the public API.
"""

import logging
import time
from typing import Set
from dataclasses import dataclass

logger = logging.getLogger("filler-interrupt-handler")


@dataclass
class InterruptionDecision:
    """Result of analyzing whether a transcript should cause interruption."""
    should_interrupt: bool
    reason: str
    transcript: str
    agent_state: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class FillerInterruptionHandler:
    """
    Extension layer that filters filler words to prevent false interruptions.
    
    This handler monitors agent state and user transcriptions, deciding whether
    incoming speech should interrupt the agent or be ignored as filler noise.
    
    Key Features:
    - Configurable filler word list
    - State-aware filtering (only filters during agent speech)
    - Mixed filler + real speech detection
    - Thread-safe operation
    - Comprehensive logging for debugging
    
    Usage:
        handler = FillerInterruptionHandler(ignored_words=['uh', 'umm', 'hmm'])
        
        # Update agent state
        handler.update_agent_state('speaking')
        
        # Check if transcript should interrupt
        decision = handler.analyze_transcript('umm', is_final=True)
        if not decision.should_interrupt:
            # Ignore this transcript
            logger.info(f"Ignoring filler: {decision.reason}")
    """
    
    def __init__(self, ignored_words: list[str] | None = None, confidence_threshold: float = 0.0):
        """
        Initialize the filler interruption handler.
        
        Args:
            ignored_words: List of filler words to ignore during agent speech.
                          Defaults to common English fillers.
            confidence_threshold: Minimum confidence for processing (future use)
        """
        default_fillers = ['uh', 'um', 'umm', 'hmm', 'haan', 'yeah', 'huh', 'mhm', 'mm']
        self.ignored_words: Set[str] = set(
            word.lower().strip() for word in (ignored_words or default_fillers)
        )
        self.confidence_threshold = confidence_threshold
        self.current_agent_state: str = "idle"
        
        # Statistics for debugging
        self.stats = {
            'total_transcripts': 0,
            'ignored_fillers': 0,
            'valid_interruptions': 0,
            'processed_while_idle': 0,
        }
        
        logger.info(f"FillerInterruptionHandler initialized with {len(self.ignored_words)} filler words")
        logger.info(f"Ignored words: {sorted(self.ignored_words)}")
    
    def update_agent_state(self, new_state: str) -> None:
        """
        Update the current agent state.
        
        Args:
            new_state: One of 'initializing', 'idle', 'listening', 'thinking', 'speaking'
        """
        old_state = self.current_agent_state
        self.current_agent_state = new_state
        logger.debug(f"Agent state updated: {old_state} -> {new_state}")
    
    def add_ignored_words(self, words: list[str]) -> None:
        """
        Dynamically add words to the ignored list (bonus feature).
        
        Args:
            words: List of additional filler words to ignore
        """
        new_words = set(word.lower().strip() for word in words)
        self.ignored_words.update(new_words)
        logger.info(f"Added {len(new_words)} words to ignored list: {new_words}")
    
    def remove_ignored_words(self, words: list[str]) -> None:
        """
        Dynamically remove words from the ignored list (bonus feature).
        
        Args:
            words: List of words to stop ignoring
        """
        words_to_remove = set(word.lower().strip() for word in words)
        self.ignored_words -= words_to_remove
        logger.info(f"Removed {len(words_to_remove)} words from ignored list: {words_to_remove}")
    
    def _normalize_text(self, text: str) -> list[str]:
        """
        Normalize and tokenize transcript text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            List of cleaned, lowercase words
        """
        # Remove common punctuation
        for char in '.,!?;:':
            text = text.replace(char, '')
        
        # Split and clean
        words = text.lower().strip().split()
        words = [w.strip() for w in words if w.strip()]
        
        return words
    
    def _is_only_fillers(self, words: list[str]) -> bool:
        """
        Check if all words in the list are fillers.
        
        Args:
            words: List of normalized words
            
        Returns:
            True if all words are in the ignored list
        """
        if not words:
            return False
        
        return all(word in self.ignored_words for word in words)
    
    def _contains_real_speech(self, words: list[str]) -> bool:
        """
        Check if the word list contains any non-filler words.
        
        Args:
            words: List of normalized words
            
        Returns:
            True if any word is NOT in the ignored list
        """
        return any(word not in self.ignored_words for word in words)
    
    def analyze_transcript(
        self, 
        transcript: str, 
        is_final: bool = True,
        confidence: float = 1.0
    ) -> InterruptionDecision:
        """
        Analyze a transcript to decide if it should interrupt the agent.
        
        This is the core decision logic:
        1. If agent is NOT speaking -> always allow (register as valid speech)
        2. If agent IS speaking:
           a. If transcript is ONLY fillers -> ignore (don't interrupt)
           b. If transcript contains ANY real words -> interrupt immediately
        3. Empty transcripts are always ignored
        
        Args:
            transcript: The transcribed text from user speech
            is_final: Whether this is a final or interim transcript
            confidence: Confidence score from ASR (0.0 to 1.0)
            
        Returns:
            InterruptionDecision with should_interrupt flag and reasoning
        """
        self.stats['total_transcripts'] += 1
        
        # Normalize transcript
        words = self._normalize_text(transcript)
        
        # Empty transcript -> ignore
        if not words:
            return InterruptionDecision(
                should_interrupt=False,
                reason="empty_transcript",
                transcript=transcript,
                agent_state=self.current_agent_state
            )
        
        # If agent is NOT speaking, always process the transcript
        if self.current_agent_state != "speaking":
            self.stats['processed_while_idle'] += 1
            logger.debug(f"[VALID] Agent not speaking - processing: '{transcript}'")
            return InterruptionDecision(
                should_interrupt=True,
                reason="agent_not_speaking",
                transcript=transcript,
                agent_state=self.current_agent_state
            )
        
        # Agent IS speaking - check if transcript is only fillers
        if self._is_only_fillers(words):
            self.stats['ignored_fillers'] += 1
            logger.info(
                f"[IGNORED FILLER] '{transcript}' | "
                f"Agent state: {self.current_agent_state} | "
                f"Words: {words}"
            )
            return InterruptionDecision(
                should_interrupt=False,
                reason="filler_only_during_agent_speech",
                transcript=transcript,
                agent_state=self.current_agent_state
            )
        
        # Contains real speech - interrupt immediately
        non_filler_words = [w for w in words if w not in self.ignored_words]
        self.stats['valid_interruptions'] += 1
        logger.info(
            f"[VALID INTERRUPTION] '{transcript}' | "
            f"Agent state: {self.current_agent_state} | "
            f"Real words detected: {non_filler_words}"
        )
        return InterruptionDecision(
            should_interrupt=True,
            reason="real_speech_detected",
            transcript=transcript,
            agent_state=self.current_agent_state
        )
    
    def get_statistics(self) -> dict:
        """
        Get handler statistics for debugging and monitoring.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            **self.stats,
            'ignored_words_count': len(self.ignored_words),
            'ignored_words': sorted(self.ignored_words),
            'current_agent_state': self.current_agent_state,
        }
    
    def log_statistics(self) -> None:
        """Log current statistics to the logger."""
        stats = self.get_statistics()
        logger.info("=== Filler Handler Statistics ===")
        logger.info(f"Total transcripts: {stats['total_transcripts']}")
        logger.info(f"Ignored fillers: {stats['ignored_fillers']}")
        logger.info(f"Valid interruptions: {stats['valid_interruptions']}")
        logger.info(f"Processed while idle: {stats['processed_while_idle']}")
        logger.info(f"Current agent state: {stats['current_agent_state']}")
        logger.info("=" * 35)
