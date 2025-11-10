"""
Test scenarios for the interruption handler.
"""

from interruption_handler import InterruptionHandler, InterruptionConfig


def test_filler_only_speech():
    """Test that filler-only speech is ignored."""
    config = InterruptionConfig.from_word_list(["uh", "umm", "hmm"])
    handler = InterruptionHandler(config)
    
    # Test cases that should be ignored
    assert handler.should_ignore_speech("uh") == True
    assert handler.should_ignore_speech("umm") == True
    assert handler.should_ignore_speech("uh umm") == True
    assert handler.should_ignore_speech("Uh, umm, hmm") == True
    
    print("✅ Filler-only speech test passed")


def test_mixed_speech():
    """Test that mixed speech with real words is not ignored."""
    config = InterruptionConfig.from_word_list(["uh", "umm", "hmm"])
    handler = InterruptionHandler(config)
    
    # Test cases that should NOT be ignored
    assert handler.should_ignore_speech("wait") == False
    assert handler.should_ignore_speech("stop") == False
    assert handler.should_ignore_speech("umm okay stop") == False
    assert handler.should_ignore_speech("uh wait a minute") == False
    
    print("✅ Mixed speech test passed")


def test_low_confidence_speech():
    """Test that low confidence speech is ignored."""
    config = InterruptionConfig.from_word_list(["uh"], confidence_threshold=0.5)
    handler = InterruptionHandler(config)
    
    # Low confidence should be ignored
    assert handler.should_ignore_speech("hello", confidence=0.3) == True
    assert handler.should_ignore_speech("hello", confidence=0.4) == True
    
    # High confidence should not be ignored
    assert handler.should_ignore_speech("hello", confidence=0.6) == False
    assert handler.should_ignore_speech("hello", confidence=0.9) == False
    
    print("✅ Low confidence speech test passed")


def test_background_murmur():
    """Test background murmur scenarios."""
    config = InterruptionConfig.from_word_list(["uh", "umm", "hmm", "yeah"], confidence_threshold=0.5)
    handler = InterruptionHandler(config)

    # Low confidence background noise - should be ignored due to low confidence
    assert handler.should_ignore_speech("hmm yeah", confidence=0.3) == True

    # Even with high confidence, if all words are fillers, should be ignored
    assert handler.should_ignore_speech("hmm yeah", confidence=0.7) == True

    # But real words with high confidence should not be ignored
    assert handler.should_ignore_speech("hello there", confidence=0.7) == False

    print("✅ Background murmur test passed")


def test_empty_and_punctuation():
    """Test empty strings and punctuation-only input."""
    config = InterruptionConfig.from_word_list(["uh"])
    handler = InterruptionHandler(config)
    
    # Empty and punctuation-only should be ignored
    assert handler.should_ignore_speech("") == True
    assert handler.should_ignore_speech("   ") == True
    assert handler.should_ignore_speech("...") == True
    assert handler.should_ignore_speech("!!!") == True
    
    print("✅ Empty and punctuation test passed")


def test_case_insensitivity():
    """Test that matching is case-insensitive."""
    config = InterruptionConfig.from_word_list(["uh", "umm"])
    handler = InterruptionHandler(config)
    
    # Different cases should all be ignored
    assert handler.should_ignore_speech("UH") == True
    assert handler.should_ignore_speech("Umm") == True
    assert handler.should_ignore_speech("UMM") == True
    assert handler.should_ignore_speech("uH uMm") == True
    
    print("✅ Case insensitivity test passed")


def test_dynamic_updates():
    """Test dynamic word list updates."""
    config = InterruptionConfig.from_word_list(
        ["uh"],
        enable_dynamic_updates=True
    )
    handler = InterruptionHandler(config)
    
    # Initially "hmm" should not be ignored (has real content)
    assert handler.should_ignore_speech("hmm") == False
    
    # Add "hmm" to ignored list
    handler.add_ignored_word("hmm")
    
    # Now "hmm" should be ignored
    assert handler.should_ignore_speech("hmm") == True
    
    # Remove "hmm" from ignored list
    handler.remove_ignored_word("hmm")
    
    # Now "hmm" should not be ignored again
    assert handler.should_ignore_speech("hmm") == False
    
    print("✅ Dynamic updates test passed")


def test_multilingual_fillers():
    """Test multilingual filler words."""
    config = InterruptionConfig.from_word_list(["uh", "umm", "haan", "hmm"])
    handler = InterruptionHandler(config)
    
    # Hindi filler
    assert handler.should_ignore_speech("haan") == True
    
    # Mixed language fillers
    assert handler.should_ignore_speech("uh haan") == True
    
    # Real Hindi word (not a filler)
    assert handler.should_ignore_speech("namaste") == False
    
    print("✅ Multilingual fillers test passed")


def run_all_tests():
    """Run all test scenarios."""
    print("\n" + "="*50)
    print("Running Interruption Handler Tests")
    print("="*50 + "\n")
    
    test_filler_only_speech()
    test_mixed_speech()
    test_low_confidence_speech()
    test_background_murmur()
    test_empty_and_punctuation()
    test_case_insensitivity()
    test_dynamic_updates()
    test_multilingual_fillers()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

