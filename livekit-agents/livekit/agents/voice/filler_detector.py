"""Filler word detection with phonetic matching support."""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Set, Dict, Any

from .phonetic_utils import PhoneticConfig, PhoneticMatcher

logger = logging.getLogger(__name__)

@dataclass
class FillerWordConfig:
    """Configuration for filler word detection.
    
    Attributes:
        filler_words: List of filler words to detect
        case_sensitive: Whether matching should be case-sensitive
        word_boundary: Whether to match whole words only
        log_filtered: Whether to log filtered filler words
        phonetic_config: Configuration for phonetic matching
        custom_phonetic_mappings: Custom phonetic mappings for specific words
    """
    filler_words: List[str] = field(default_factory=lambda: ["uh", "umm", "hmm", "haan"])
    case_sensitive: bool = False
    word_boundary: bool = True
    log_filtered: bool = True
    phonetic_config: PhoneticConfig = field(default_factory=PhoneticConfig)
    custom_phonetic_mappings: Dict[str, List[str]] = field(default_factory=dict)

class FillerWordDetector:
    """Detects and handles filler words in speech transcripts with phonetic matching."""
    
    def __init__(self, config: Optional[FillerWordConfig] = None):
        """Initialize the filler word detector."""
        self.config = config or FillerWordConfig()
        self._filler_words = set(self.config.filler_words)
        self._filler_cache = {}
        self._phonetic_matcher = PhoneticMatcher(self.config.phonetic_config)
        self._update_pattern()
        
    def _update_pattern(self):
        """Update the regex pattern based on current filler words."""
        if not self._filler_words:
            self._pattern = None
            return
            
        # Sort by length in descending order to match longer phrases first
        sorted_fillers = sorted(self._filler_words, key=len, reverse=True)
        
        # Escape special regex characters and create word boundary patterns
        escaped = [re.escape(f) for f in sorted_fillers]
        
        # Create a pattern that matches whole words only
        pattern = r'(?<!\w)(?:' + '|'.join(escaped) + r')(?!\w)'
        
        # Compile with case-insensitive flag if needed
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        self._pattern = re.compile(pattern, flags)
        
    def _normalize_word(self, word: str) -> str:
        """Normalize a word by removing repeated characters."""
        if not word:
            return word
            
        word_lower = word.lower()
        
        # Special case for 'hmm' variations
        if word_lower.startswith('h'):
            # For 'h' words, keep up to 2 'm's
            normalized = re.sub(r'h(m{2,})', 'hmm', word_lower)
            # For 'hm' with no vowels, make it 'hmm'
            normalized = re.sub(r'^hm$', 'hmm', normalized)
            return normalized
            
        # Special case for 'um' variations
        if word_lower.startswith('um') or word_lower.startswith('uh'):
            # For 'ummm' -> 'um', 'uhhh' -> 'uh'
            normalized = re.sub(r'^(u[hm])([hm]{2,})$', r'\1', word_lower)
            if normalized != word_lower:
                return normalized
            # For 'umm' -> 'um', 'uhh' -> 'uh'
            normalized = re.sub(r'^(u[hm])([hm])$', r'\1', word_lower)
            if normalized != word_lower:
                return normalized
            
        # For other words, normalize repeated characters to max 2
        normalized = []
        prev_char = None
        repeat_count = 0
        
        for char in word_lower:
            if char == prev_char:
                repeat_count += 1
                if repeat_count < 2:  # Allow up to 2 repeated characters
                    normalized.append(char)
            else:
                normalized.append(char)
                prev_char = char
                repeat_count = 0
                
        return ''.join(normalized)
        
    def is_filler(self, word: str) -> bool:
        """Check if a word is a filler word.
        
        Args:
            word: The word to check
            
        Returns:
            True if the word is a filler word, False otherwise
        """
        if not word:
            return False
            
        # Check cache first
        cache_key = word.lower() if not self.config.case_sensitive else word
        if cache_key in self._filler_cache:
            return self._filler_cache[cache_key]
        
        # Normalize the word (remove excessive repeated characters)
        normalized_word = self._normalize_word(word)
        
        # Check for exact match with any filler word (both original and normalized)
        for filler in self._filler_words:
            # For case-insensitive matching
            if ((not self.config.case_sensitive and 
                 (word.lower() == filler.lower() or normalized_word.lower() == filler.lower())) or 
                (self.config.case_sensitive and (word == filler or normalized_word == filler))):
                self._filler_cache[cache_key] = True
                if self.config.log_filtered:
                    logger.debug(f"Exact match: '{word}' is a filler word")
                return True
        
        # If no exact match and phonetic matching is enabled, try phonetic matching
        if self.config.phonetic_config.enabled:
            for filler in self._filler_words:
                # Skip multi-word fillers for phonetic matching
                if ' ' in filler:
                    continue
                    
                # Check if the words are similar in length (prevent false positives)
                min_len = min(len(word), len(filler))
                max_len = max(len(word), len(filler))
                if min_len == 0 or max_len / min_len > 1.5:  # Made more strict
                    continue
                    
                # Additional check: first letter should be the same for better accuracy
                if word[0].lower() != filler[0].lower():
                    continue
                    
                # For custom mappings, only check against the specific mapped variations
                if (hasattr(self.config.phonetic_config, 'custom_mappings') and 
                    self.config.phonetic_config.custom_mappings and 
                    filler in self.config.phonetic_config.custom_mappings):
                    # Only check against explicitly mapped variations
                    mapped_variations = self.config.phonetic_config.custom_mappings[filler]
                    if word.lower() not in [v.lower() for v in mapped_variations]:
                        continue
                
                # Check phonetic match with both original and normalized word
                if (self._phonetic_matcher.is_match(word, filler) or 
                    self._phonetic_matcher.is_match(normalized_word, filler)):
                    # Additional check: for custom mappings, only allow exact matches
                    if (hasattr(self.config.phonetic_config, 'custom_mappings') and 
                        self.config.phonetic_config.custom_mappings and 
                        filler in self.config.phonetic_config.custom_mappings):
                        if word.lower() not in [v.lower() for v in self.config.phonetic_config.custom_mappings[filler]]:
                            continue
                    
                    self._filler_cache[cache_key] = True
                    if self.config.log_filtered:
                        logger.debug(f"Phonetic match: '{word}' matches filler word: '{filler}'")
                    return True
        
        self._filler_cache[cache_key] = False
        return False
    
    def is_filler_only(self, text: str) -> bool:
        """Check if the text contains only filler words.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text contains only filler words, False otherwise
        """
        if not text.strip():
            return False
            
        # First check if the entire text is a filler (for multi-word fillers)
        cleaned_text = text.strip().lower()
        for filler in self._filler_words:
            # Check both with and without normalization
            if ' ' in filler and (cleaned_text == filler.lower() or 
                                 cleaned_text.replace(',', ' ').strip() == filler.lower()):
                return True
        
        # Try with normalized text (remove all non-word chars except spaces)
        normalized_text = re.sub(r'[^\w\s]', ' ', cleaned_text)  # Replace punctuation with spaces
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()  # Normalize spaces
        
        # Check if the entire normalized text is a multi-word filler
        if normalized_text in [f.lower() for f in self._filler_words if ' ' in f]:
            return True
            
        # Check if the text is a comma-separated list of fillers
        comma_separated = [w.strip() for w in cleaned_text.split(',') if w.strip()]
        if comma_separated and all(w in [f.lower() for f in self._filler_words] for w in comma_separated):
            return True
                
        # Otherwise, split into words and check each one
        words = re.findall(r'\b\w+\b', normalized_text)
        return len(words) > 0 and all(self.is_filler(word) for word in words)
    
    def get_fillers(self, text: str) -> List[str]:
        """Get all filler words found in the text.
        
        Args:
            text: The text to search for filler words
            
        Returns:
            List of filler words found in the text
        """
        if not text.strip() or not self._filler_words:
            return []
            
        # Find all potential filler words in the text
        matches = []
        for word in re.findall(r'\b\w+(?: \w+)*\b', text):
            if self.is_filler(word):
                matches.append(word)
            else:
                # Check if any part of a multi-word phrase is a filler
                for phrase in re.split(r'\s+', word):
                    if self.is_filler(phrase):
                        matches.append(phrase)
        
        return matches
    
    def update_config(self, config: FillerWordConfig):
        """Update the filler word configuration.
        
        Args:
            config: New configuration for filler word detection
        """
        self.config = config
        self._filler_words = set(self.config.filler_words)  # Reset filler words
        self._phonetic_matcher = PhoneticMatcher(self.config.phonetic_config)
        self._update_pattern()
        self._filler_cache.clear()  # Clear cache on config update
        
        # Re-apply any custom phonetic mappings
        if hasattr(self.config, 'custom_phonetic_mappings') and self.config.custom_phonetic_mappings:
            for word, forms in self.config.custom_phonetic_mappings.items():
                self._phonetic_matcher.add_custom_mapping(word, forms)
    
    def add_filler_word(self, word: str):
        """Add a filler word to the detector.
        
        Args:
            word: The filler word to add
        """
        # Add the word in the correct case based on config
        word_to_add = word.lower() if not self.config.case_sensitive else word
        self._filler_words.add(word_to_add)
        self._update_pattern()
        # Clear cache since we've added a new filler word
        self._filler_cache.clear()
    
    def remove_filler_word(self, word: str):
        """Remove a filler word from the detector.
        
        Args:
            word: The filler word to remove
        """
        if not hasattr(self, '_filler_words') or not word:
            return
            
        word_to_remove = word.lower() if not self.config.case_sensitive else word
        self._filler_words = {w for w in self._filler_words if w != word_to_remove}
        self._update_pattern()
        self._filler_cache.clear()  # Clear cache since we've removed a filler word
