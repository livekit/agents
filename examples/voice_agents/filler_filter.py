"""
Intelligent Filler Word Filter for LiveKit Voice Agents
========================================================

This module provides smart interruption handling by distinguishing between:
- Filler words (uh, umm, hmm, haan) that should be ignored during agent speech
- Genuine interruptions (stop, wait, etc.) that should immediately interrupt

Author: NSUT Team
Assignment: SalesCode AI - LiveKit Voice Interruption Challenge
"""

import logging
import os
from typing import Set, List, Dict
from datetime import datetime

logger = logging.getLogger("filler-filter")


class FillerWordFilter:
    """
    Intelligent filter that prevents false interruptions from filler words
    while allowing genuine user interruptions to proceed.
    
    Key Features:
    - Configurable filler word list
    - Confidence threshold filtering for background noise
    - Real-time agent state tracking
    - Comprehensive session statistics
    - Dynamic word list updates (bonus feature)
    """
    
    def __init__(self, ignored_words: List[str] = None, min_confidence: float = 0.6):
        """
        Initialize the filler word filter.
        
        Args:
            ignored_words: List of filler words to ignore during agent speech
            min_confidence: Minimum ASR confidence threshold (0.0-1.0)
        """
        # Load filler words from parameter or environment
        if ignored_words is None:
            env_words = os.getenv('IGNORED_WORDS', 'uh,umm,hmm,haan,um,er,ah')
            ignored_words = [w.strip().lower() for w in env_words.split(',')]
        
        self.ignored_words: Set[str] = set(ignored_words)
        self.agent_is_speaking: bool = False
        self.min_confidence: float = min_confidence
        
        # Comprehensive statistics tracking
        self.stats: Dict[str, int] = {
            'total_transcriptions': 0,
            'ignored_fillers': 0,
            'valid_interruptions': 0,
            'low_confidence_filtered': 0,
            'agent_quiet_processed': 0,
            'empty_transcriptions': 0
        }
        
        # Track specific examples for debugging
        self.filler_examples: List[str] = []
        self.valid_interrupt_examples: List[str] = []
        
        # Session metadata
        self.session_start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info("🚀 FillerWordFilter Initialized")
        logger.info(f"   Ignored words: {list(self.ignored_words)}")
        logger.info(f"   Min confidence: {self.min_confidence}")
        logger.info(f"   Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
    
    def set_agent_speaking(self, is_speaking: bool):
        """
        Update the agent's speaking state.
        
        Args:
            is_speaking: True if agent is currently speaking, False otherwise
        """
        if self.agent_is_speaking != is_speaking:
            self.agent_is_speaking = is_speaking
            state_emoji = "🗣️" if is_speaking else "👂"
            state_text = "SPEAKING" if is_speaking else "LISTENING"
            logger.debug(f"{state_emoji} Agent state changed: {state_text}")
    
    def should_allow_interruption(self, text: str, confidence: float = 1.0) -> bool:
        """
        Determine if the transcribed text should trigger an interruption.
        
        Core Logic:
        1. If agent is NOT speaking → Allow all speech (return True)
        2. If agent IS speaking:
           - Filter low confidence (likely noise) → Deny (return False)
           - Check if contains only filler words → Deny (return False)
           - If contains ANY non-filler words → Allow (return True)
        
        Args:
            text: The transcribed user speech
            confidence: ASR confidence score (0.0-1.0)
            
        Returns:
            True if interruption should be allowed, False if should be ignored
        """
        self.stats['total_transcriptions'] += 1
        
        # Handle empty or whitespace-only transcriptions
        if not text or not text.strip():
            self.stats['empty_transcriptions'] += 1
            logger.debug("[EMPTY] Ignored empty transcription")
            return False
        
        # Normalize text
        text_clean = text.lower().strip()
        words = text_clean.split()
        
        # Filter 1: Low confidence (background noise, mumbling)
        if confidence < self.min_confidence:
            self.stats['low_confidence_filtered'] += 1
            logger.info(f"🔇 [LOW_CONFIDENCE] Filtered (conf={confidence:.2f}): '{text}'")
            return False
        
        # Case 1: Agent is NOT speaking - accept ALL speech
        if not self.agent_is_speaking:
            self.stats['agent_quiet_processed'] += 1
            logger.debug(f"👂 [AGENT_QUIET] Processing: '{text}'")
            return True
        
        # Case 2: Agent IS speaking - apply filler filtering
        non_filler_words = [word for word in words if word not in self.ignored_words]
        
        # If contains ANY non-filler content → Valid interruption
        if non_filler_words:
            self.stats['valid_interruptions'] += 1
            
            # Store example for reporting (max 5)
            if len(self.valid_interrupt_examples) < 5:
                self.valid_interrupt_examples.append(text)
            
            logger.info(
                f"✅ [VALID_INTERRUPT] Allowing interruption\n"
                f"   Full text: '{text}'\n"
                f"   Non-filler words: {non_filler_words}\n"
                f"   Confidence: {confidence:.2f}"
            )
            return True
        
        # Only filler words → Block interruption
        self.stats['ignored_fillers'] += 1
        
        # Store example for reporting (max 5)
        if len(self.filler_examples) < 5:
            self.filler_examples.append(text)
        
        logger.info(
            f"🚫 [FILLER_BLOCKED] Preventing false interruption\n"
            f"   Filler words only: '{text}'\n"
            f"   Confidence: {confidence:.2f}\n"
            f"   → Agent will continue speaking"
        )
        return False
    
    def update_ignored_words(self, new_words: List[str]):
        """
        Dynamically update the list of ignored words during runtime.
        
        Bonus Feature: Allows runtime reconfiguration of filler words.
        
        Args:
            new_words: New list of words to ignore
        """
        old_words = self.ignored_words.copy()
        self.ignored_words = set(w.strip().lower() for w in new_words)
        
        logger.info("🔄 Ignored words updated")
        logger.info(f"   Before: {list(old_words)}")
        logger.info(f"   After:  {list(self.ignored_words)}")
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive session statistics.
        
        Returns:
            Dictionary containing all statistics and calculated metrics
        """
        total = self.stats['total_transcriptions']
        
        return {
            **self.stats,
            'filler_block_rate': (self.stats['ignored_fillers'] / max(1, total)) * 100,
            'valid_interrupt_rate': (self.stats['valid_interruptions'] / max(1, total)) * 100,
            'filter_efficiency': (
                (self.stats['ignored_fillers'] + self.stats['low_confidence_filtered']) / 
                max(1, total)
            ) * 100,
            'session_duration': str(datetime.now() - self.session_start_time),
        }
    
    def print_stats(self):
        """
        Print comprehensive session statistics to the log.
        """
        stats = self.get_stats()
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 FILLER FILTER SESSION STATISTICS")
        logger.info("=" * 70)
        
        # Core metrics
        logger.info("Core Metrics:")
        logger.info(f"  📝 Total transcriptions: {stats['total_transcriptions']}")
        logger.info(f"  ✅ Valid interruptions allowed: {stats['valid_interruptions']}")
        logger.info(f"  🚫 Fillers blocked: {stats['ignored_fillers']}")
        logger.info(f"  🔇 Low confidence filtered: {stats['low_confidence_filtered']}")
        logger.info(f"  👂 Processed (agent quiet): {stats['agent_quiet_processed']}")
        logger.info(f"  ⚠️  Empty transcriptions: {stats['empty_transcriptions']}")
        
        logger.info("")
        
        # Calculated rates
        logger.info("Performance Rates:")
        logger.info(f"  Filler block rate: {stats['filler_block_rate']:.1f}%")
        logger.info(f"  Valid interrupt rate: {stats['valid_interrupt_rate']:.1f}%")
        logger.info(f"  Overall filter efficiency: {stats['filter_efficiency']:.1f}%")
        
        logger.info("")
        
        # Session info
        logger.info("Session Info:")
        logger.info(f"  Duration: {stats['session_duration']}")
        logger.info(f"  Ignored words: {list(self.ignored_words)}")
        logger.info(f"  Min confidence: {self.min_confidence}")
        
        # Examples
        if self.filler_examples:
            logger.info("")
            logger.info("Example Fillers Blocked:")
            for example in self.filler_examples:
                logger.info(f"  • '{example}'")
        
        if self.valid_interrupt_examples:
            logger.info("")
            logger.info("Example Valid Interruptions:")
            for example in self.valid_interrupt_examples:
                logger.info(f"  • '{example}'")
        
        logger.info("=" * 70 + "\n")
