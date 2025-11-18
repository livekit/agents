"""Unit tests for filler word suppression.

This module tests the filler word suppression functionality including:
- Basic word suppression
- Multi-language pattern matching
- Confidence thresholds
- Advanced suppression logic with 6 algorithms
- Real-world conversation scenarios

Author: Niranjani Sharma
Date: November 19, 2025
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "livekit-agents",
        "livekit",
        "agents",
        "voice",
    ),
)

from suppression_config import SuppressionConfig


def test_basic_suppression():
    """Test basic word suppression."""
    config = SuppressionConfig()
    config._suppression_words = {"uh", "umm", "hmm"}
    
    assert config.is_suppressed_word("uh")
    assert config.is_suppressed_word("UMM")
    assert not config.is_suppressed_word("hello")
    print("✓ Basic suppression test passed")


def test_regex_pattern_matching():
    """Test multi-language regex patterns."""
    config = SuppressionConfig()
    
    # English pattern
    assert config.matches_pattern("uh umm er", "en")
    assert config.matches_pattern("hmm", "en")
    assert not config.matches_pattern("hello there", "en")
    
    # Hindi pattern
    assert config.matches_pattern("haan arey", "hi")
    assert not config.matches_pattern("namaste", "hi")
    
    # Spanish pattern
    assert config.matches_pattern("eh este", "es")
    
    # French pattern
    assert config.matches_pattern("euh ben", "fr")
    
    print("✓ Regex pattern matching test passed")


def test_confidence_threshold():
    """Test confidence-based suppression."""
    config = SuppressionConfig()
    config._min_confidence = 0.7
    
    assert config.get_min_confidence() == 0.7
    
    # Update threshold
    config.set_min_confidence(0.8)
    assert config.get_min_confidence() == 0.8
    
    # Test invalid threshold
    try:
        config.set_min_confidence(1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Confidence threshold test passed")


def test_dynamic_updates():
    """Test adding/removing suppression words."""
    config = SuppressionConfig()
    initial_count = len(config.get_suppression_words())
    
    # Add words
    config.add_suppression_words(["yaar", "bas", "theek"])
    assert "yaar" in config.get_suppression_words()
    assert "bas" in config.get_suppression_words()
    assert "theek" in config.get_suppression_words()
    
    # Remove words
    config.remove_suppression_words(["yaar"])
    assert "yaar" not in config.get_suppression_words()
    assert "bas" in config.get_suppression_words()
    
    # Clear all
    config.clear_suppression_words()
    assert len(config.get_suppression_words()) == 0
    
    print("✓ Dynamic updates test passed")


def test_multilanguage_support():
    """Test all supported languages."""
    config = SuppressionConfig(enable_patterns=True)
    
    test_cases = [
        ("uh umm", "en", True),
        ("haan yaar", "hi", True),
        ("este pues", "es", True),
        ("euh alors", "fr", True),
        ("äh also", "de", True),
        ("ano eto", "ja", True),
        ("嗯 那个", "zh", True),
        ("né então", "pt", True),
        ("ehm allora", "it", True),
        ("그 저", "ko", True),
        # Non-filler text
        ("hello world", "en", False),
        ("namaste dost", "hi", False),
        ("hola amigo", "es", False),
    ]
    
    for text, lang, expected in test_cases:
        result = config.matches_pattern(text, lang)
        assert result == expected, f"Failed for {text} ({lang}): expected {expected}, got {result}"
    
    print("✓ Multi-language support test passed")


def test_case_insensitivity():
    """Test that suppression is case-insensitive."""
    config = SuppressionConfig()
    config._suppression_words = {"uh", "umm"}
    
    assert config.is_suppressed_word("UH")
    assert config.is_suppressed_word("Umm")
    assert config.is_suppressed_word("uMM")
    assert config.is_suppressed_word("uh")
    
    print("✓ Case insensitivity test passed")


def test_empty_and_whitespace():
    """Test handling of empty and whitespace-only text."""
    config = SuppressionConfig()
    
    # Empty text should not match patterns
    assert not config.matches_pattern("", "en")
    assert not config.matches_pattern("   ", "en")
    
    print("✓ Empty/whitespace test passed")


def test_advanced_suppression_logic():
    """Test the advanced 6-algorithm suppression logic."""
    config = SuppressionConfig(detection_mode="balanced", min_filler_ratio=0.7)
    
    # Test Algorithm 1: Confidence thresholding
    should_suppress, reason = config.should_suppress_advanced("hello", 0.3, "en")
    assert should_suppress, "Should suppress low confidence"
    assert "confidence" in reason.lower()
    
    # Test Algorithm 2: Empty text
    should_suppress, reason = config.should_suppress_advanced("   ", 0.9, "en")
    assert should_suppress, "Should suppress empty text"
    
    # Test Algorithm 3: Pure fillers
    should_suppress, reason = config.should_suppress_advanced("uh um er", 0.9, "en")
    assert should_suppress, "Should suppress pure fillers"
    
    # Test Algorithm 6: Single acknowledgments
    should_suppress, reason = config.should_suppress_advanced("okay", 0.9, "en")
    assert not should_suppress, "Should allow single acknowledgment"
    
    should_suppress, reason = config.should_suppress_advanced("yeah", 0.9, "en")
    assert not should_suppress, "Should allow 'yeah'"
    
    # Test real questions with fillers
    should_suppress, reason = config.should_suppress_advanced(
        "uh can you help me", 0.9, "en"
    )
    assert not should_suppress, "Should allow questions with fillers"
    
    # Test acknowledgment in longer phrase
    should_suppress, reason = config.should_suppress_advanced("okay wait", 0.9, "en")
    assert not should_suppress, "Should allow acknowledgment with content"
    
    print("✓ Advanced suppression logic test passed")


def test_detection_modes():
    """Test different detection modes: strict, balanced, lenient."""
    # Strict mode (100% fillers only)
    strict_config = SuppressionConfig(detection_mode="strict")
    should_suppress, _ = strict_config.should_suppress_advanced("uh um er", 0.9, "en")
    assert should_suppress, "Strict mode should suppress 100% fillers"
    
    should_suppress, _ = strict_config.should_suppress_advanced("uh hello", 0.9, "en")
    assert not should_suppress, "Strict mode should not suppress <100% fillers"
    
    # Balanced mode (>=70% fillers)
    balanced_config = SuppressionConfig(detection_mode="balanced", min_filler_ratio=0.7)
    should_suppress, _ = balanced_config.should_suppress_advanced("uh um er ah", 0.9, "en")
    assert should_suppress, "Balanced mode should suppress 100% fillers"
    
    # Lenient mode (>=80% fillers)
    lenient_config = SuppressionConfig(detection_mode="lenient")
    should_suppress, _ = lenient_config.should_suppress_advanced("uh um er ah oh", 0.9, "en")
    assert should_suppress, "Lenient mode should suppress 100% fillers"
    
    should_suppress, _ = lenient_config.should_suppress_advanced("uh hello", 0.9, "en")
    assert not should_suppress, "Lenient mode should allow <80% fillers"
    
    print("✓ Detection modes test passed")


def test_real_world_scenarios():
    """Test real-world conversation scenarios."""
    config = SuppressionConfig(detection_mode="balanced", min_filler_ratio=0.7)
    
    test_cases = [
        # (text, confidence, expected_suppress, description)
        ("uh I have a question", 0.92, False, "Question with filler prefix"),
        ("um can you help me", 0.91, False, "Request with filler"),
        ("okay", 0.88, False, "Simple acknowledgment"),
        ("yeah sure", 0.89, False, "Positive response"),
        ("uh", 0.85, True, "Single filler"),
        ("um er uh", 0.85, True, "Multiple fillers only"),
        ("", 0.90, True, "Empty string"),
        ("hello there", 0.92, False, "Clear greeting"),
        ("I uh need help", 0.90, False, "Filler mid-sentence"),
    ]
    
    for text, confidence, expected_suppress, description in test_cases:
        should_suppress, reason = config.should_suppress_advanced(text, confidence, "en")
        assert should_suppress == expected_suppress, (
            f"Failed for '{text}' ({description}): "
            f"expected {expected_suppress}, got {should_suppress}. Reason: {reason}"
        )
    
    print("✓ Real-world scenarios test passed")


def test_acknowledgment_words():
    """Test acknowledgment word handling."""
    config = SuppressionConfig()
    
    # Test all acknowledgment words are allowed when used alone
    acknowledgment_words = [
        "okay", "ok", "yeah", "yes", "yep", "sure", "right", "correct",
        "alright", "fine", "good", "great", "perfect", "excellent",
        "no", "nope", "nah", "never", "absolutely", "definitely"
    ]
    
    for word in acknowledgment_words:
        should_suppress, reason = config.should_suppress_advanced(word, 0.9, "en")
        assert not should_suppress, (
            f"Acknowledgment word '{word}' should not be suppressed. Reason: {reason}"
        )
    
    print("✓ Acknowledgment words test passed")


def test_filler_ratio_calculation():
    """Test filler ratio calculation."""
    config = SuppressionConfig()
    
    # 100% fillers
    ratio = config.calculate_filler_ratio("uh um er")
    assert ratio == 1.0, f"Expected 1.0, got {ratio}"
    
    # 50% fillers
    ratio = config.calculate_filler_ratio("uh hello")
    assert 0.4 < ratio < 0.6, f"Expected ~0.5, got {ratio}"
    
    # 0% fillers
    ratio = config.calculate_filler_ratio("hello world")
    assert ratio == 0.0, f"Expected 0.0, got {ratio}"
    
    # Empty text
    ratio = config.calculate_filler_ratio("")
    assert ratio == 0.0, f"Expected 0.0 for empty, got {ratio}"
    
    print("✓ Filler ratio calculation test passed")


def run_all_tests():
    """Run all suppression tests."""
    print("=" * 70)
    print("Running Filler Suppression Tests")
    print("=" * 70)
    print()
    
    # Core functionality tests
    test_basic_suppression()
    test_regex_pattern_matching()
    test_confidence_threshold()
    test_dynamic_updates()
    test_multilanguage_support()
    test_case_insensitivity()
    test_empty_and_whitespace()
    
    # Advanced feature tests
    test_advanced_suppression_logic()
    test_detection_modes()
    test_real_world_scenarios()
    test_acknowledgment_words()
    test_filler_ratio_calculation()
    
    print()
    print("=" * 70)
    print("✅ All 12 tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
