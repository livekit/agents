"""
LiveKit Intelligent Interruption Handler
Filters filler words during agent speech while preserving genuine interruptions
"""

import asyncio
import logging
import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InterruptionType(Enum):
    """Types of interruptions detected"""
    FILLER = "filler"
    GENUINE = "genuine"
    MIXED = "mixed"


@dataclass
class InterruptionEvent:
    """Represents a detected interruption"""
    transcript: str
    type: InterruptionType
    confidence: float
    timestamp: float
    agent_was_speaking: bool


class IntelligentInterruptionHandler:
    """
    Handles intelligent filtering of user interruptions in LiveKit voice agents.
    
    Distinguishes between filler words (uh, umm, hmm) and genuine interruptions
    based on agent speaking state and transcription content.
    """
    
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
        enable_logging: bool = True,
        case_sensitive: bool = False
    ):
        """
        Initialize the interruption handler.
        
        Args:
            ignored_words: List of filler words to ignore when agent is speaking
            confidence_threshold: Minimum ASR confidence to process (0.0 to 1.0)
            enable_logging: Whether to log interruption events
            case_sensitive: Whether word matching should be case-sensitive
        """
        # Default filler words
        default_fillers = [
            'uh', 'uhh', 'um', 'umm', 'hmm', 'hmmm', 'err', 'ah', 'ahh',
            'haan', 'haan', 'mhmm', 'uh-huh', 'mm-hmm', 'yeah'
        ]
        
        self.ignored_words = set(ignored_words or default_fillers)
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self.case_sensitive = case_sensitive
        
        # State tracking
        self._agent_is_speaking = False
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_interruptions': 0,
            'filler_ignored': 0,
            'genuine_processed': 0,
            'mixed_processed': 0
        }
        
        logger.info(f"Initialized InterruptionHandler with {len(self.ignored_words)} filler words")
        logger.debug(f"Filler words: {sorted(self.ignored_words)}")
    
    def set_agent_speaking(self, is_speaking: bool):
        """Update the agent's speaking state."""
        self._agent_is_speaking = is_speaking
        if self.enable_logging:
            logger.debug(f"Agent speaking state changed: {is_speaking}")
    
    def is_agent_speaking(self) -> bool:
        """Check if agent is currently speaking"""
        return self._agent_is_speaking
    
    def add_ignored_words(self, words: List[str]):
        """Dynamically add words to the ignored list."""
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        self.ignored_words.update(words)
        logger.info(f"Added {len(words)} words to ignored list")
    
    def remove_ignored_words(self, words: List[str]):
        """Dynamically remove words from the ignored list."""
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        self.ignored_words.difference_update(words)
        logger.info(f"Removed {len(words)} words from ignored list")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract individual words from text, handling punctuation."""
        text = re.sub(r'[^\w\s-]', '', text)
        words = text.split()
        
        if not self.case_sensitive:
            words = [w.lower() for w in words]
        
        return words
    
    def _is_only_fillers(self, text: str) -> bool:
        """Check if text contains only filler words."""
        words = self._extract_words(text)
        if not words:
            return True
        
        return all(word in self.ignored_words for word in words)
    
    def _contains_genuine_speech(self, text: str) -> bool:
        """Check if text contains any non-filler words."""
        words = self._extract_words(text)
        return any(word not in self.ignored_words for word in words)
    
    def classify_interruption(
        self,
        transcript: str,
        confidence: float
    ) -> InterruptionType:
        """Classify an interruption as filler, genuine, or mixed."""
        if not transcript or not transcript.strip():
            return InterruptionType.FILLER
        
        normalized = self._normalize_text(transcript)
        
        if self._is_only_fillers(normalized):
            return InterruptionType.FILLER
        
        has_genuine = self._contains_genuine_speech(normalized)
        has_filler = any(word in self.ignored_words 
                        for word in self._extract_words(normalized))
        
        if has_genuine and has_filler:
            return InterruptionType.MIXED
        elif has_genuine:
            return InterruptionType.GENUINE
        else:
            return InterruptionType.FILLER
    
    async def should_process_interruption(
        self,
        transcript: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> tuple[bool, InterruptionEvent]:
        """
        Determine if an interruption should be processed based on current state.
        
        Returns:
            Tuple of (should_process, event_details)
        """
        async with self._lock:
            self.stats['total_interruptions'] += 1
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                if self.enable_logging:
                    logger.debug(f"Low confidence ({confidence:.2f}), ignoring: '{transcript}'")
                return False, InterruptionEvent(
                    transcript=transcript,
                    type=InterruptionType.FILLER,
                    confidence=confidence,
                    timestamp=timestamp or asyncio.get_event_loop().time(),
                    agent_was_speaking=self._agent_is_speaking
                )
            
            # Classify the interruption
            interrupt_type = self.classify_interruption(transcript, confidence)
            
            event = InterruptionEvent(
                transcript=transcript,
                type=interrupt_type,
                confidence=confidence,
                timestamp=timestamp or asyncio.get_event_loop().time(),
                agent_was_speaking=self._agent_is_speaking
            )
            
            # Decision logic
            if not self._agent_is_speaking:
                # Agent is quiet - process all speech
                if self.enable_logging:
                    logger.info(f"✓ Agent quiet, processing: '{transcript}' ({interrupt_type.value})")
                self.stats['genuine_processed'] += 1
                return True, event
            
            # Agent is speaking - filter based on type
            if interrupt_type == InterruptionType.FILLER:
                if self.enable_logging:
                    logger.info(f"✗ Filler ignored (agent speaking): '{transcript}'")
                self.stats['filler_ignored'] += 1
                return False, event
            
            elif interrupt_type in (InterruptionType.GENUINE, InterruptionType.MIXED):
                if self.enable_logging:
                    logger.warning(f"⚠ INTERRUPTION detected (agent speaking): '{transcript}' ({interrupt_type.value})")
                if interrupt_type == InterruptionType.MIXED:
                    self.stats['mixed_processed'] += 1
                else:
                    self.stats['genuine_processed'] += 1
                return True, event
            
            return False, event
    
    def get_stats(self) -> dict:
        """Get handler statistics"""
        return {
            **self.stats,
            'ignored_words_count': len(self.ignored_words),
            'agent_currently_speaking': self._agent_is_speaking
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_interruptions': 0,
            'filler_ignored': 0,
            'genuine_processed': 0,
            'mixed_processed': 0
        }
        logger.info("Statistics reset")