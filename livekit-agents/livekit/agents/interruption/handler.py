"""
LiveKit Intelligent Interruption Handler
Filters filler words during agent speech while allowing real interruptions
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Set

from livekit import rtc
from livekit.agents import utils, stt

logger = logging.getLogger(__name__)


class InterruptionType(Enum):
    """Classification of user speech interruptions"""
    FILLER_ONLY = "filler_only"
    REAL_INTERRUPTION = "real_interruption"
    LOW_CONFIDENCE = "low_confidence"
    VALID_SPEECH = "valid_speech"


@dataclass
class InterruptionConfig:
    """Configuration for interruption handling"""
    ignored_words: Set[str]
    confidence_threshold: float = 0.6
    min_word_duration: float = 0.2
    log_all_events: bool = True
    allow_runtime_updates: bool = True


@dataclass
class InterruptionEvent:
    """Records an interruption event for logging/debugging"""
    timestamp: float
    text: str
    classification: InterruptionType
    confidence: float
    agent_was_speaking: bool
    action_taken: str


class IntelligentInterruptionHandler:
    """
    Handles voice interruptions intelligently by filtering filler words
    when the agent is speaking while allowing real interruptions through.
    """
    
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
        log_all_events: bool = True,
        allow_runtime_updates: bool = True
    ):
        """
        Initialize the interruption handler.
        
        Args:
            ignored_words: List of filler words to ignore during agent speech
            confidence_threshold: Minimum confidence for valid interruptions
            log_all_events: Whether to log all interruption events
            allow_runtime_updates: Allow updating ignored words at runtime
        """
        if ignored_words is None:
            ignored_words = [
                'uh', 'umm', 'hmm', 'hm', 'mm', 'mhmm',
                'haan', 'han', 'ha', 'ah', 'eh', 'er'
            ]
        
        self._config = InterruptionConfig(
            ignored_words=set(word.lower() for word in ignored_words),
            confidence_threshold=confidence_threshold,
            log_all_events=log_all_events,
            allow_runtime_updates=allow_runtime_updates
        )
        
        self._agent_speaking = False
        self._interruption_history: List[InterruptionEvent] = []
        self._lock = asyncio.Lock()
        self._on_valid_interruption: Optional[Callable] = None
        
        logger.info(
            f"Initialized InterruptionHandler with ignored words: {self._config.ignored_words}"
        )
    
    def set_agent_speaking(self, is_speaking: bool) -> None:
        """Update the agent's speaking state"""
        self._agent_speaking = is_speaking
        logger.debug(f"Agent speaking state changed to: {is_speaking}")
    
    def set_interruption_callback(self, callback: Callable) -> None:
        """Set callback to invoke on valid interruptions"""
        self._on_valid_interruption = callback
    
    async def update_ignored_words(self, words: List[str], append: bool = True) -> None:
        """
        Dynamically update the list of ignored words.
        
        Args:
            words: New list of words
            append: If True, add to existing list; if False, replace
        """
        if not self._config.allow_runtime_updates:
            logger.warning("Runtime updates not allowed in current configuration")
            return
        
        async with self._lock:
            normalized_words = set(word.lower() for word in words)
            if append:
                self._config.ignored_words.update(normalized_words)
                logger.info(f"Added ignored words: {normalized_words}")
            else:
                self._config.ignored_words = normalized_words
                logger.info(f"Replaced ignored words with: {normalized_words}")
    
    def _classify_interruption(
        self,
        text: str,
        confidence: float,
        agent_speaking: bool
    ) -> InterruptionType:
        """
        Classify the type of interruption based on content and context.
        
        Args:
            text: Transcribed text
            confidence: ASR confidence score
            agent_speaking: Whether agent is currently speaking
            
        Returns:
            InterruptionType classification
        """
        if confidence < self._config.confidence_threshold and agent_speaking:
            return InterruptionType.LOW_CONFIDENCE
        
        words = text.lower().strip().split()
        
        if not words:
            return InterruptionType.LOW_CONFIDENCE
        
        non_filler_words = [w for w in words if w not in self._config.ignored_words]
        
        if not non_filler_words:
            if agent_speaking:
                return InterruptionType.FILLER_ONLY
            else:
                return InterruptionType.VALID_SPEECH
        else:
            return InterruptionType.REAL_INTERRUPTION
    
    async def process_transcript(
        self,
        text: str,
        confidence: float,
        is_final: bool = True
    ) -> bool:
        """
        Process a transcript and determine if it should interrupt the agent.
        
        Args:
            text: Transcribed text
            confidence: ASR confidence score
            is_final: Whether this is a final transcript
            
        Returns:
            True if agent should be interrupted, False otherwise
        """
        async with self._lock:
            classification = self._classify_interruption(
                text=text,
                confidence=confidence,
                agent_speaking=self._agent_speaking
            )
            
            should_interrupt = classification == InterruptionType.REAL_INTERRUPTION
            
            if classification == InterruptionType.FILLER_ONLY:
                action = "ignored (filler only)"
            elif classification == InterruptionType.LOW_CONFIDENCE:
                action = "ignored (low confidence)"
            elif classification == InterruptionType.REAL_INTERRUPTION:
                action = "interrupted agent"
            else:
                action = "registered as valid speech"
            
            event = InterruptionEvent(
                timestamp=time.time(),
                text=text,
                classification=classification,
                confidence=confidence,
                agent_was_speaking=self._agent_speaking,
                action_taken=action
            )
            self._interruption_history.append(event)
            
            if self._config.log_all_events or should_interrupt:
                logger.info(
                    f"Interruption Event: '{text}' | "
                    f"Type: {classification.value} | "
                    f"Confidence: {confidence:.2f} | "
                    f"Agent Speaking: {self._agent_speaking} | "
                    f"Action: {action}"
                )
            
            if should_interrupt and self._on_valid_interruption:
                await self._on_valid_interruption(event)
            
            return should_interrupt
    
    def get_interruption_history(
        self,
        limit: Optional[int] = None
    ) -> List[InterruptionEvent]:
        """Get recent interruption history for debugging"""
        if limit:
            return self._interruption_history[-limit:]
        return self._interruption_history.copy()
    
    def get_statistics(self) -> dict:
        """Get statistics about interruption handling"""
        if not self._interruption_history:
            return {
                'total_events': 0,
                'by_type': {},
                'ignored_words': list(self._config.ignored_words),
                'confidence_threshold': self._config.confidence_threshold
            }
        
        total = len(self._interruption_history)
        by_type = {}
        for event in self._interruption_history:
            type_name = event.classification.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            'total_events': total,
            'by_type': by_type,
            'ignored_words': list(self._config.ignored_words),
            'confidence_threshold': self._config.confidence_threshold
        }


class LiveKitInterruptionWrapper:
    """
    Wrapper that integrates IntelligentInterruptionHandler with LiveKit agent.
    """
    
    def __init__(
        self,
        handler: IntelligentInterruptionHandler,
        original_interrupt_callback: Optional[Callable] = None
    ):
        self._handler = handler
        self._original_callback = original_interrupt_callback
    
    async def on_stt_event(self, event: stt.SpeechEvent) -> None:
        """Handle STT events from LiveKit."""
        if not hasattr(event, 'alternatives') or not event.alternatives:
            return
        
        alternative = event.alternatives[0]
        text = alternative.text
        confidence = getattr(alternative, 'confidence', 1.0)
        is_final = event.is_final
        
        should_interrupt = await self._handler.process_transcript(
            text=text,
            confidence=confidence,
            is_final=is_final
        )
        
        if should_interrupt and self._original_callback:
            await self._original_callback(event)
    
    async def on_agent_speech_start(self) -> None:
        """Notify handler that agent started speaking"""
        self._handler.set_agent_speaking(True)
    
    async def on_agent_speech_end(self) -> None:
        """Notify handler that agent stopped speaking"""
        self._handler.set_agent_speaking(False)