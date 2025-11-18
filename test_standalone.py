"""
Standalone test for FillerFilter - No dependencies required
This tests the core logic without requiring the full LiveKit package.

Author: Raghav
Date: November 18, 2025
"""


class SimpleFillerFilter:
    """Standalone version of FillerFilter for testing."""
    
    def __init__(self, ignored_words=None, min_confidence_threshold=0.5):
        self.min_confidence_threshold = min_confidence_threshold
        if ignored_words:
            self.ignored_words = [w.strip().lower() for w in ignored_words]
        else:
            self.ignored_words = ["uh", "umm", "hmm", "haan", "mm", "mhm", "er", "ah", "oh"]
    
    def is_filler_only(self, text, confidence=1.0, agent_is_speaking=False):
        # Low confidence check
        if confidence < self.min_confidence_threshold:
            return True
        
        # Empty text check
        text_cleaned = text.strip()
        if not text_cleaned:
            return True
        
        # Normalize words
        words = self._normalize_words(text_cleaned)
        if not words:
            return True
        
        # Check if ALL words are fillers
        for word in words:
            if word not in self.ignored_words:
                return False
        
        return True
    
    def _normalize_words(self, text):
        text_cleaned = text.lower()
        for punct in ".,!?;:\"'":
            text_cleaned = text_cleaned.replace(punct, " ")
        words = [w.strip() for w in text_cleaned.split() if w.strip()]
        return words


def test_filler_filter():
    """Test the FillerFilter with various scenarios."""
    
    print("=" * 80)
    print("STANDALONE FILLER FILTER TEST SUITE")
    print("=" * 80)
    print()
    
    # Initialize filter
    filter_obj = SimpleFillerFilter()
    print(f"✓ Initialized FillerFilter")
    print(f"  Ignored words: {filter_obj.ignored_words}")
    print(f"  Confidence threshold: {filter_obj.min_confidence_threshold}")
    print()
    
    # Test cases
    test_cases = [
        ("Simple filler", "umm", 0.8, True, True),
        ("Valid interruption", "stop", 0.9, True, False),
        ("Mixed filler + valid", "umm stop", 0.85, True, False),
        ("Multiple fillers", "hmm haan", 0.7, True, True),
        ("Low confidence", "something", 0.3, True, True),
        ("Empty text", "", 1.0, True, True),
        ("Whitespace", "   ", 1.0, True, True),
        ("Filler with punct", "umm,", 0.8, True, True),
        ("Valid command", "wait a minute", 0.95, True, False),
        ("High conf valid", "hello", 0.99, True, False),
    ]
    
    passed = 0
    failed = 0
    
    for i, (name, text, confidence, agent_speaking, expected) in enumerate(test_cases, 1):
        result = filter_obj.is_filler_only(text, confidence, agent_speaking)
        
        status = "✓ PASS" if result == expected else "✗ FAIL"
        emoji = "🟢" if result == expected else "🔴"
        
        print(f"{emoji} {status} - Test {i}: {name}")
        print(f"   Input: '{text}' (conf: {confidence})")
        print(f"   Expected: {expected}, Got: {result}")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"   ⚠️  MISMATCH!")
        print()
    
    # Summary
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "STANDALONE FILLER FILTER TEST" + " " * 34 + "║")
    print("║" + " " * 28 + "by Raghav" + " " * 42 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    success = test_filler_filter()
    
    print("\n")
    print("=" * 80)
    if success:
        print("🎉🎉🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
        print()
        print("The filler filter implementation is working correctly!")
        print()
        print("Next steps:")
        print("1. Install LiveKit agents: cd livekit-agents && pip install -e .")
        print("2. Run full test suite: python test_filler_filter.py")
        print("3. Try the example agent: python examples/filler_filter_example.py")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)
    print("\n")
