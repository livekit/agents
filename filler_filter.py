"""
Filler Interruption Filter Module
Intelligently filters filler words to prevent false interruptions

Author: Guttula Viswa Venkata Yashwanth
"""

import logging
import re
from typing import List, Dict, Set
from datetime import datetime

logger = logging.getLogger("filler-filter")


class FillerInterruptionFilter:
    """
    Intelligent filter for filler word interruptions.
    
    Core Logic:
    - Ignores filler words ONLY when the agent is currently speaking
    - Registers those same words as valid user speech when agent is quiet
    - Uses confidence thresholding for additional filtering
    """
    
    def __init__(
        self, 
        ignored_words: List[str] = None, 
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the filler filter.
        
        Args:
            ignored_words: List of filler words to filter (e.g., ['uh', 'umm'])
            confidence_threshold: Min STT confidence to register (0.0-1.0)
        """
        # Convert to set for O(1) lookup, normalize to lowercase
        self.ignored_words: Set[str] = set(
            w.lower().strip() for w in (ignored_words or [])
        )
        self.confidence_threshold = confidence_threshold
        self.agent_is_speaking = False
        
        # Statistics tracking
        self.ignored_count = 0
        self.valid_count = 0
        self.ignored_log: List[Dict] = []
        self.valid_log: List[Dict] = []
        
        # Log initialization
        logger.info(f"🎯 Filler filter initialized")
        logger.info(f"   Filler words ({len(self.ignored_words)}): {', '.join(sorted(self.ignored_words))}")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
    
    def set_agent_speaking(self, is_speaking: bool):
        """
        Update the agent's speaking state.
        
        Args:
            is_speaking: True if agent is currently speaking, False otherwise
        """
        self.agent_is_speaking = is_speaking
        state = "🗣️ SPEAKING" if is_speaking else "👂 LISTENING"
        logger.debug(f"Agent state changed: {state}")
    
    def should_ignore_interruption(self, text: str, confidence: float = 1.0) -> bool:
        """
        Determine if transcription should be ignored as a filler interruption.
        
        Decision logic:
        1. If agent NOT speaking → Don't ignore (register all speech)
        2. If agent IS speaking:
           - Check if text contains only filler words
           - Check if confidence is below threshold
           - Ignore if BOTH conditions are true
        
        Args:
            text: Transcribed text from the user
            confidence: STT confidence score (0.0-1.0)
        
        Returns:
            True if should be ignored, False if valid speech
        """
        # RULE 1: Never ignore when agent is not speaking
        if not self.agent_is_speaking:
            return False
        
        # Clean and tokenize the text
        cleaned_text = self._clean_text(text)
        words = cleaned_text.split()
        
        # If no words detected, likely noise - ignore
        if not words:
            return True
        
        # Check how many words are NOT fillers
        non_filler_words = [w for w in words if w not in self.ignored_words]
        
        # RULE 2: If ALL words are fillers
        if len(non_filler_words) == 0:
            # RULE 2a: Low confidence filler → Ignore
            if confidence < self.confidence_threshold:
                return True
            # RULE 2b: Single high-confidence filler → Still likely ignore
            if len(words) == 1:
                return True
            # RULE 2c: Multiple high-confidence fillers → Process (might be intentional)
            return False
        
        # RULE 3: If ANY meaningful content exists → Don't ignore
        return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for comparison.
        
        Args:
            text: Raw transcribed text
        
        Returns:
            Cleaned lowercase text without punctuation
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def log_ignored_interruption(self, text: str, confidence: float):
        """Record an ignored filler interruption for statistics"""
        self.ignored_count += 1
        self.ignored_log.append({
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'confidence': confidence,
            'agent_speaking': self.agent_is_speaking
        })
    
    def log_valid_interruption(self, text: str, confidence: float):
        """Record a valid interruption for statistics"""
        self.valid_count += 1
        self.valid_log.append({
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'confidence': confidence,
            'agent_speaking': self.agent_is_speaking
        })
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive filtering statistics.
        
        Returns:
            Dictionary with counts, rates, and detailed logs
        """
        total = self.ignored_count + self.valid_count
        filter_rate = self.ignored_count / total if total > 0 else 0
        
        return {
            'ignored_fillers': self.ignored_count,
            'valid_interruptions': self.valid_count,
            'total_events': total,
            'filter_rate': filter_rate,
            'ignored_log': self.ignored_log,
            'valid_log': self.valid_log
        }
    
    def add_filler_word(self, word: str):
        """
        Dynamically add a filler word to the ignore list.
        
        Args:
            word: Filler word to add
        """
        normalized = word.lower().strip()
        self.ignored_words.add(normalized)
        logger.info(f"➕ Added filler word: '{normalized}'")
    
    def remove_filler_word(self, word: str):
        """
        Remove a filler word from the ignore list.
        
        Args:
            word: Filler word to remove
        """
        normalized = word.lower().strip()
        self.ignored_words.discard(normalized)
        logger.info(f"➖ Removed filler word: '{normalized}'")
    
    def get_ignored_words(self) -> Set[str]:
        """Get current set of ignored filler words"""
        return self.ignored_words.copy()
