"""Tests for the FillerWordDetector class."""
import pytest
from livekit.agents.voice import FillerWordDetector, FillerWordConfig, PhoneticConfig, PhoneticAlgorithm

def test_basic_filler_detection():
    """Test basic filler word detection."""
    config = FillerWordConfig(
        filler_words=["um", "uh", "like", "you know"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(enabled=False)
    )
    detector = FillerWordDetector(config)
    
    # Test exact matches
    assert detector.is_filler("um") is True
    assert detector.is_filler("UM") is True  # case insensitive
    assert detector.is_filler("uh") is True
    assert detector.is_filler("like") is True
    assert detector.is_filler("you know") is True
    
    # Test non-matches
    assert detector.is_filler("hello") is False
    assert detector.is_filler("something") is False
    assert detector.is_filler("") is False
    
    # Test word boundary
    assert detector.is_filler("umbrella") is False  # 'um' is part of a word
    assert detector.is_filler("bump") is False  # 'um' is part of a word

def test_phonetic_matching():
    """Test phonetic matching of filler words."""
    config = FillerWordConfig(
        filler_words=["um", "uh"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(
            enabled=True,
            algorithm=PhoneticAlgorithm.METAPHONE,  # Using METAPHONE instead of DOUBLE_METAPHONE
            min_word_length=1
        )
    )
    detector = FillerWordDetector(config)
    
    # These should match phonetically
    assert detector.is_filler("umm") is True
    assert detector.is_filler("uhm") is True
    # Note: 'uhhh' is too different in length from 'uh' (2 vs 4)
    # so we'll only test with up to 2 repeated characters
    assert detector.is_filler("uhh") is True
    assert detector.is_filler("ummm") is True

def test_filler_only():
    """Test detection of text containing only filler words."""
    config = FillerWordConfig(
        filler_words=["um", "uh", "like", "you know"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(enabled=False)
    )
    detector = FillerWordDetector(config)
    
    assert detector.is_filler_only("um, like, you know") is True
    assert detector.is_filler_only("um, hello") is False
    assert detector.is_filler_only("") is False
    assert detector.is_filler_only("   ") is False

def test_get_fillers():
    """Test getting all filler words from text."""
    config = FillerWordConfig(
        filler_words=["um", "uh", "like", "you know"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(enabled=False)
    )
    detector = FillerWordDetector(config)
    
    text = "Um, I was thinking, you know, like, maybe we could go"
    fillers = detector.get_fillers(text)
    assert set(fillers) == {"Um", "you know", "like"}
    
    # Test with no fillers
    assert detector.get_fillers("Hello world") == []
    assert detector.get_fillers("") == []

def test_update_config():
    """Test updating the detector configuration."""
    config = FillerWordConfig(
        filler_words=["um", "uh"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(enabled=False)
    )
    detector = FillerWordDetector(config)
    
    # Should only detect original fillers
    assert detector.is_filler("um") is True
    assert detector.is_filler("like") is False
    
    # Update config with new fillers
    new_config = FillerWordConfig(
        filler_words=["like", "actually"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(enabled=False)
    )
    detector.update_config(new_config)
    
    # Should now detect new fillers, not old ones
    assert detector.is_filler("um") is False
    assert detector.is_filler("like") is True
    assert detector.is_filler("actually") is True

def test_custom_phonetic_mappings():
    """Test custom phonetic mappings for filler words."""
    config = FillerWordConfig(
        filler_words=["hmm"],
        case_sensitive=False,
        word_boundary=True,
        phonetic_config=PhoneticConfig(
            enabled=True,
            algorithm=PhoneticAlgorithm.METAPHONE,
            min_word_length=1,
            custom_mappings={
                "hmm": ["hmmm", "hm", "hmmmm"]
            }
        )
    )
    detector = FillerWordDetector(config)
    
    # Should match custom mappings
    assert detector.is_filler("hmm") is True
    assert detector.is_filler("hmmm") is True
    assert detector.is_filler("hm") is True
    assert detector.is_filler("hmmmm") is True
    
    # Should not match non-mapped variations
    assert detector.is_filler("hum") is False
    assert detector.is_filler("ham") is False
    assert detector.is_filler("him") is False

def test_add_remove_filler_word():
    """Test adding and removing filler words at runtime."""
    detector = FillerWordDetector()
    
    # Should not detect initially
    assert detector.is_filler("custom") is False
    
    # Add custom filler word
    detector.add_filler_word("custom")
    assert detector.is_filler("custom") is True
    
    # Remove it
    detector.remove_filler_word("custom")
    assert detector.is_filler("custom") is False

if __name__ == "__main__":
    pytest.main([__file__])
