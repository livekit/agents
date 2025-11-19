import asyncio
import logging
from typing import Optional, List, Set
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InterruptionEvent:
    """Log entry for interruption events"""
    timestamp: datetime
    transcription: str
    confidence: float
    agent_speaking: bool
    action_taken: str  # 'ignored', 'stopped', 'registered'
    reason: str

class IntelligentInterruptionHandler:
    """
    Handles intelligent filtering of user interruptions.
    Distinguishes between filler words and genuine interruptions.
    """
    
    def __init__(self, config):
        self.config = config
        self.agent_speaking = False
        self.event_log: List[InterruptionEvent] = []
        
        # Convert ignored words to set for O(1) lookup
        self.ignored_words_set: Set[str] = {
            word.lower() for word in config.ignored_words
        }
        
        # Lock for thread-safe state management
        self._lock = asyncio.Lock()
    
    async def set_agent_speaking(self, speaking: bool):
        """Update agent speaking state (thread-safe)"""
        async with self._lock:
            self.agent_speaking = speaking
            if self.config.debug_mode:
                logger.info(f"Agent speaking state: {speaking}")
    
    async def should_interrupt(
        self, 
        transcription: str, 
        confidence: float = 1.0
    ) -> bool:
        """
        Determine if transcription should interrupt the agent.
        
        Returns:
            True if agent should stop (genuine interruption)
            False if should be ignored (filler or low confidence)
        """
        async with self._lock:
            # Clean and normalize transcription
            text = transcription.strip().lower()
            words = text.split()
            
            # Log the event
            event = InterruptionEvent(
                timestamp=datetime.now(),
                transcription=transcription,
                confidence=confidence,
                agent_speaking=self.agent_speaking,
                action_taken='pending',
                reason=''
            )
            
            # Rule 1: If agent is not speaking, always register as valid speech
            if not self.agent_speaking:
                event.action_taken = 'registered'
                event.reason = 'Agent not speaking'
                self.event_log.append(event)
                if self.config.debug_mode:
                    logger.info(f"✓ Registered: '{transcription}' (agent quiet)")
                return True
            
            # Rule 2: Check confidence threshold
            if confidence < self.config.confidence_threshold:
                event.action_taken = 'ignored'
                event.reason = f'Low confidence ({confidence:.2f})'
                self.event_log.append(event)
                if self.config.debug_mode:
                    logger.info(f"✗ Ignored: '{transcription}' (low confidence)")
                return False
            
            # Rule 3: Check if transcription contains ONLY filler words
            non_filler_words = [
                word for word in words 
                if word not in self.ignored_words_set
            ]
            
            if len(non_filler_words) == 0:
                # Only fillers detected
                event.action_taken = 'ignored'
                event.reason = 'Only filler words detected'
                self.event_log.append(event)
                if self.config.debug_mode:
                    logger.info(f"✗ Ignored: '{transcription}' (filler only)")
                return False
            
            # Rule 4: Contains non-filler words = genuine interruption
            event.action_taken = 'stopped'
            event.reason = f'Contains command words: {non_filler_words}'
            self.event_log.append(event)
            if self.config.debug_mode:
                logger.info(f"⚠ Stopping: '{transcription}' (genuine interruption)")
            return True
    
    def update_ignored_words(self, new_words: List[str]):
        """Dynamically update the ignored words list (bonus feature)"""
        self.ignored_words_set.update(word.lower() for word in new_words)
        if self.config.debug_mode:
            logger.info(f"Updated ignored words: {self.ignored_words_set}")
    
    def get_event_log(self) -> List[InterruptionEvent]:
        """Return interruption event log for debugging"""
        return self.event_log.copy()
    
    def clear_log(self):
        """Clear event log"""
        self.event_log.clear()
