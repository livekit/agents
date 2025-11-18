"""Phonetic matching utilities for filler word detection."""
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, ClassVar
import re
import logging

logger = logging.getLogger(__name__)

class PhoneticAlgorithm:
    """Supported phonetic algorithms."""
    SOUNDEX: ClassVar[str] = "soundex"
    METAPHONE: ClassVar[str] = "metaphone"
    DOUBLE_METAPHONE: ClassVar[str] = "double_metaphone"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all supported phonetic algorithms."""
        return [cls.SOUNDEX, cls.METAPHONE, cls.DOUBLE_METAPHONE]

@dataclass
class PhoneticConfig:
    """Configuration for phonetic matching.
    
    Attributes:
        enabled: Whether phonetic matching is enabled
        algorithm: The phonetic algorithm to use
        min_word_length: Minimum word length to apply phonetic matching
        include_original: Whether to include the original word in phonetic forms
        custom_mappings: Custom phonetic mappings for specific words
    """
    def __init__(
        self,
        enabled: bool = True,
        algorithm: str = PhoneticAlgorithm.METAPHONE,
        min_word_length: int = 2,
        include_original: bool = True,
        custom_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        self.enabled = enabled
        self.algorithm = algorithm
        self.min_word_length = min_word_length
        self.include_original = include_original
        self.custom_mappings = custom_mappings if custom_mappings is not None else {}

class PhoneticMatcher:
    """Handles phonetic matching of words using various algorithms."""
    
    def __init__(self, config: Optional[PhoneticConfig] = None):
        """Initialize the phonetic matcher with the given configuration.
        
        Args:
            config: Phonetic matching configuration
        """
        self.config = config or PhoneticConfig()
        self._cache: Dict[str, Set[str]] = {}
        self._algorithm = None
        self._init_algorithm()
    
    def _init_algorithm(self):
        """Initialize the phonetic algorithm based on config."""
        if not self.config.enabled:
            return
            
        try:
            if self.config.algorithm == PhoneticAlgorithm.SOUNDEX:
                from jellyfish import soundex
                self._algorithm = soundex
            elif self.config.algorithm == PhoneticAlgorithm.METAPHONE:
                from jellyfish import metaphone
                self._algorithm = metaphone
            elif self.config.algorithm == PhoneticAlgorithm.DOUBLE_METAPHONE:
                # Fall back to metaphone if double_metaphone is not available
                try:
                    from jellyfish import double_metaphone
                    self._algorithm = double_metaphone
                except ImportError:
                    from jellyfish import metaphone
                    self._algorithm = metaphone
                    logger.warning("double_metaphone not available, falling back to metaphone")
            else:
                raise ValueError(f"Unsupported phonetic algorithm: {self.config.algorithm}")
        except ImportError as e:
            logger.warning("jellyfish not installed, phonetic matching will be disabled: %s", e)
            self.config.enabled = False
    
    def get_phonetic(self, word: str) -> Set[str]:
        """Get phonetic representation(s) of a word.
        
        Args:
            word: The word to get phonetic forms for
            
        Returns:
            Set of phonetic forms for the word
        """
        if not self.config.enabled or not word or len(word) < self.config.min_word_length:
            return set()
            
        if word in self._cache:
            return self._cache[word]
            
        phonetic_forms = set()
        
        # Add original word if configured
        if self.config.include_original:
            phonetic_forms.add(word.lower())
        
        # Add phonetic forms if algorithm is available
        if self._algorithm is not None:
            try:
                if self.config.algorithm == PhoneticAlgorithm.DOUBLE_METAPHONE:
                    # Handle case where double_metaphone might not be available
                    try:
                        dm1, dm2 = self._algorithm(word)
                        if dm1:
                            phonetic_forms.add(dm1.lower())
                        if dm2:
                            phonetic_forms.add(dm2.lower())
                    except (TypeError, ValueError):
                        # Fall back to single metaphone if double_metaphone fails
                        phonetic_form = self._algorithm(word)
                        if phonetic_form:
                            phonetic_forms.add(phonetic_form.lower())
                else:
                    phonetic_form = self._algorithm(word)
                    if phonetic_form:
                        phonetic_forms.add(phonetic_form.lower())
            except Exception as e:
                logger.warning("Error applying phonetic algorithm: %s", e)
        
        # Add custom mappings
        if self.config.custom_mappings and word.lower() in self.config.custom_mappings:
            for mapped_form in self.config.custom_mappings[word.lower()]:
                phonetic_forms.add(mapped_form.lower())
        
        self._cache[word] = phonetic_forms
        return phonetic_forms
    
    def is_match(self, word1: str, word2: str) -> bool:
        """Check if two words match phonetically.
        
        Args:
            word1: First word to compare
            word2: Second word to compare
            
        Returns:
            True if the words match phonetically, False otherwise
        """
        if not self.config.enabled:
            return word1.lower() == word2.lower()
            
        if not word1 or not word2:
            return False
            
        # If words are the same (case-insensitive), they match
        if word1.lower() == word2.lower():
            return True
            
        # If one word is much longer than the other, they probably don't match
        min_len = min(len(word1), len(word2))
        max_len = max(len(word1), len(word2))
        if min_len == 0 or max_len / min_len > 2:
            return False
            
        # Get phonetic forms
        forms1 = self.get_phonetic(word1)
        forms2 = self.get_phonetic(word2)
        
        # If either word has no phonetic forms, do direct comparison
        if not forms1 or not forms2:
            return word1.lower() == word2.lower()
            
        # Check for any matching phonetic forms
        return len(forms1.intersection(forms2)) > 0
    
    def add_custom_mapping(self, word: str, phonetic_forms: List[str]):
        """Add custom phonetic mappings for a word.
        
        Args:
            word: The word to add mappings for
            phonetic_forms: List of phonetic forms for the word
        """
        if not self.config.custom_mappings:
            self.config.custom_mappings = {}
            
        word_lower = word.lower()
        if word_lower not in self.config.custom_mappings:
            self.config.custom_mappings[word_lower] = []
            
        self.config.custom_mappings[word_lower].extend(
            form.lower() for form in phonetic_forms
        )
        
        # Invalidate cache for this word
        if word_lower in self._cache:
            del self._cache[word_lower]

# Add __all__ to explicitly export public API
__all__ = ['PhoneticAlgorithm', 'PhoneticConfig', 'PhoneticMatcher']