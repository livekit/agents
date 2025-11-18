"""
Utility functions for text processing and filler detection
"""
import re
import logging
from typing import List, Set


logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_words(text: str) -> List[str]:
    """
    Extract individual words from text
    
    Args:
        text: Input text
    
    Returns:
        List of words
    """
    normalized = normalize_text(text)
    return normalized.split()


def is_filler_only(text: str, filler_words: Set[str]) -> bool:
    """
    Check if text contains only filler words
    
    Args:
        text: Input text
        filler_words: Set of filler words to check against
    
    Returns:
        True if text contains only filler words, False otherwise
    """
    if not text or not text.strip():
        return True
    
    normalized = normalize_text(text)
    
    # Check for multi-word fillers first (e.g., "you know", "haan ji")
    for filler in sorted(filler_words, key=len, reverse=True):
        if ' ' in filler:  # Multi-word filler
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(filler) + r'\b'
            normalized = re.sub(pattern, '', normalized)
    
    # Extract remaining words
    words = extract_words(normalized)
    
    # Check if all remaining words are fillers
    return all(word in filler_words for word in words) if words else True


def contains_command(text: str, command_words: Set[str] = None) -> bool:
    """
    Check if text contains command words that should interrupt the agent
    
    Args:
        text: Input text
        command_words: Set of command words (default: common interrupt commands)
    
    Returns:
        True if text contains command words, False otherwise
    """
    if command_words is None:
        # Default command words that indicate real interruption
        command_words = {
            'wait', 'stop', 'hold', 'pause', 'no', 'yes',
            'okay', 'ok', 'continue', 'go', 'next', 'back',
            'repeat', 'again', 'what', 'when', 'where', 'who', 'how', 'why'
        }
    
    words = extract_words(text)
    return any(word in command_words for word in words)


def should_filter(
    text: str,
    confidence: float,
    is_agent_speaking: bool,
    filler_words: Set[str],
    confidence_threshold: float = 0.3,
    debug: bool = False
) -> bool:
    """
    Determine if a transcription should be filtered
    
    Args:
        text: Transcribed text
        confidence: Transcription confidence score
        is_agent_speaking: Whether agent is currently speaking
        filler_words: Set of filler words
        confidence_threshold: Minimum confidence to consider
        debug: Enable debug logging
    
    Returns:
        True if the transcription should be filtered, False otherwise
    """
    # If agent is not speaking, never filter
    if not is_agent_speaking:
        if debug:
            logger.debug(f"Not filtering (agent not speaking): '{text}'")
        return False
    
    # If confidence is too low, filter it
    if confidence > 0 and confidence < confidence_threshold:
        if debug:
            logger.debug(f"Filtering (low confidence {confidence:.2f}): '{text}'")
        return True
    
    # If text contains command words, don't filter
    if contains_command(text):
        if debug:
            logger.debug(f"Not filtering (contains command): '{text}'")
        return False
    
    # If text is only fillers, filter it
    if is_filler_only(text, filler_words):
        if debug:
            logger.debug(f"Filtering (filler only): '{text}'")
        return True
    
    # Otherwise, don't filter
    if debug:
        logger.debug(f"Not filtering (contains content): '{text}'")
    return False
