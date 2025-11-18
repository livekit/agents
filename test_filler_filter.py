"""
Test script for Filler Filter functionality
This demonstrates how the filler filter works with various test cases.

Author: Raghav
Date: November 18, 2025
"""

import asyncio
import sys
import os

# Add the livekit-agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'livekit-agents'))

from livekit.agents.voice.filler_filter import FillerFilter


def test_filler_filter():
    """Test the FillerFilter with various scenarios."""
    
    print("=" * 80)
    print("FILLER FILTER TEST SUITE")
    print("=" * 80)
    print()
    
    # Initialize filter with default settings
    filter_default = FillerFilter()
    print(f"✓ Initialized FillerFilter with default words: {filter_default.get_ignored_words()}")
    print(f"✓ Confidence threshold: {filter_default._min_confidence_threshold}")
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Simple filler (agent speaking)",
            "text": "umm",
            "confidence": 0.8,
            "agent_speaking": True,
            "expected": True,
            "reason": "Single filler word should be ignored"
        },
        {
            "name": "Test 2: Filler when agent silent",
            "text": "umm",
            "confidence": 0.8,
            "agent_speaking": False,
            "expected": True,
            "reason": "Filler is still filler even when agent not speaking"
        },
        {
            "name": "Test 3: Valid interruption",
            "text": "stop",
            "confidence": 0.9,
            "agent_speaking": True,
            "expected": False,
            "reason": "'stop' is not a filler word"
        },
        {
            "name": "Test 4: Mixed filler + valid word",
            "text": "umm stop",
            "confidence": 0.85,
            "agent_speaking": True,
            "expected": False,
            "reason": "Contains 'stop' which is not a filler"
        },
        {
            "name": "Test 5: Multiple fillers",
            "text": "hmm haan",
            "confidence": 0.7,
            "agent_speaking": True,
            "expected": True,
            "reason": "Both words are fillers"
        },
        {
            "name": "Test 6: Low confidence murmur",
            "text": "something unclear",
            "confidence": 0.3,
            "agent_speaking": True,
            "expected": True,
            "reason": "Below confidence threshold (0.5)"
        },
        {
            "name": "Test 7: Empty text",
            "text": "",
            "confidence": 1.0,
            "agent_speaking": True,
            "expected": True,
            "reason": "Empty text should be treated as filler"
        },
        {
            "name": "Test 8: Whitespace only",
            "text": "   ",
            "confidence": 1.0,
            "agent_speaking": True,
            "expected": True,
            "reason": "Whitespace-only should be treated as filler"
        },
        {
            "name": "Test 9: Filler with punctuation",
            "text": "umm,",
            "confidence": 0.8,
            "agent_speaking": True,
            "expected": True,
            "reason": "Punctuation should be stripped"
        },
        {
            "name": "Test 10: Valid command",
            "text": "wait a minute",
            "confidence": 0.95,
            "agent_speaking": True,
            "expected": False,
            "reason": "Valid speech with multiple non-filler words"
        },
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = filter_default.is_filler_only(
            test["text"],
            test["confidence"],
            test["agent_speaking"]
        )
        
        status = "✓ PASS" if result == test["expected"] else "✗ FAIL"
        emoji = "🟢" if result == test["expected"] else "🔴"
        
        print(f"{emoji} {status} - {test['name']}")
        print(f"   Input: '{test['text']}' (confidence: {test['confidence']})")
        print(f"   Expected: {test['expected']}, Got: {result}")
        print(f"   Reason: {test['reason']}")
        
        if result == test["expected"]:
            passed += 1
        else:
            failed += 1
            print(f"   ⚠️  MISMATCH!")
        
        print()
    
    # Summary
    print("=" * 80)
    print(f"TEST SUMMARY: {passed}/{len(test_cases)} passed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    print("=" * 80)
    print()
    
    return failed == 0


async def test_runtime_updates():
    """Test runtime updates to the filler filter."""
    
    print("=" * 80)
    print("TESTING RUNTIME UPDATES")
    print("=" * 80)
    print()
    
    # Create filter with custom words
    filter_custom = FillerFilter(ignored_words=["uh", "umm", "hmm"])
    print(f"Initial filler words: {filter_custom.get_ignored_words()}")
    print()
    
    # Test 1: Add words
    print("Test 1: Adding new filler words...")
    await filter_custom.add_ignored_words(["haan", "arey"])
    print(f"After adding: {filter_custom.get_ignored_words()}")
    assert "haan" in filter_custom.get_ignored_words()
    assert "arey" in filter_custom.get_ignored_words()
    print("✓ Words added successfully")
    print()
    
    # Test 2: Remove words
    print("Test 2: Removing filler words...")
    await filter_custom.remove_ignored_words(["hmm"])
    print(f"After removing 'hmm': {filter_custom.get_ignored_words()}")
    assert "hmm" not in filter_custom.get_ignored_words()
    print("✓ Words removed successfully")
    print()
    
    # Test 3: Update entire list
    print("Test 3: Updating entire list...")
    await filter_custom.update_ignored_words(["new1", "new2", "new3"])
    print(f"After update: {filter_custom.get_ignored_words()}")
    assert filter_custom.get_ignored_words() == ["new1", "new2", "new3"]
    print("✓ List updated successfully")
    print()
    
    # Test 4: Change confidence threshold
    print("Test 4: Changing confidence threshold...")
    filter_custom.set_confidence_threshold(0.7)
    print(f"New threshold: {filter_custom._min_confidence_threshold}")
    assert filter_custom._min_confidence_threshold == 0.7
    print("✓ Threshold updated successfully")
    print()
    
    print("=" * 80)
    print("🎉 All runtime update tests passed!")
    print("=" * 80)
    print()


def test_environment_variable():
    """Test loading filler words from environment variable."""
    
    print("=" * 80)
    print("TESTING ENVIRONMENT VARIABLE CONFIGURATION")
    print("=" * 80)
    print()
    
    # Set environment variable
    os.environ["IGNORED_WORDS"] = "custom1,custom2,custom3"
    
    # Create new filter (should load from env)
    filter_env = FillerFilter()
    
    print(f"Environment variable: {os.environ.get('IGNORED_WORDS')}")
    print(f"Loaded filler words: {filter_env.get_ignored_words()}")
    
    expected = ["custom1", "custom2", "custom3"]
    if filter_env.get_ignored_words() == expected:
        print("✓ Environment variable loaded correctly")
        print()
        print("=" * 80)
        print("🎉 Environment variable test passed!")
        print("=" * 80)
        print()
        return True
    else:
        print("✗ Environment variable not loaded correctly")
        print()
        return False


def test_logging_output():
    """Test that logging works correctly."""
    
    print("=" * 80)
    print("TESTING LOGGING OUTPUT")
    print("=" * 80)
    print()
    
    import logging
    
    # Configure logging to see debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    filter_test = FillerFilter(ignored_words=["uh", "umm"])
    
    print("Test 1: Check low confidence logging...")
    filter_test.is_filler_only("some text", confidence=0.2, agent_is_speaking=True)
    print()
    
    print("Test 2: Check filler-only detection logging...")
    filter_test.is_filler_only("uh umm", confidence=0.8, agent_is_speaking=True)
    print()
    
    print("✓ Logging tests completed (check output above)")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "FILLER FILTER TEST SUITE" + " " * 34 + "║")
    print("║" + " " * 25 + "by Raghav" + " " * 45 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run all tests
    success = True
    
    # 1. Basic functionality tests
    if not test_filler_filter():
        success = False
    
    # 2. Runtime update tests
    asyncio.run(test_runtime_updates())
    
    # 3. Environment variable test
    if not test_environment_variable():
        success = False
    
    # 4. Logging test
    test_logging_output()
    
    # Final summary
    print("\n")
    print("=" * 80)
    if success:
        print("🎉🎉🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
    else:
        print("⚠️  SOME TESTS FAILED - Please review output above")
    print("=" * 80)
    print("\n")
    
    sys.exit(0 if success else 1)
