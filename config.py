import os
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class InterruptionConfig:
    """Configuration for intelligent interruption handling"""
    
    # Filler words to ignore (expandable)
    ignored_words: List[str] = None
    
    # Confidence threshold for ASR (0-1)
    confidence_threshold: float = 0.6
    
    # Enable debug logging
    debug_mode: bool = False
    
    # Language-specific fillers
    language_fillers: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.ignored_words is None:
            self.ignored_words = [
                'uh', 'um', 'umm', 'hmm', 'haan', 
                'mhm', 'aha', 'uhh', 'err', 'ah'
            ]
        
        if self.language_fillers is None:
            self.language_fillers = {
                'en': ['uh', 'um', 'hmm'],
                'hi': ['haan', 'hmm', 'achha'],
                'es': ['eh', 'este', 'pues']
            }
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        ignored = os.getenv('IGNORED_WORDS', '').split(',')
        ignored = [w.strip().lower() for w in ignored if w.strip()]
        
        return cls(
            ignored_words=ignored if ignored else None,
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.6')),
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        )
