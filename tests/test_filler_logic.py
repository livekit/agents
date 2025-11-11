"""
Standalone test for filler filter logic without dependencies.

This tests the core filtering algorithm without requiring LiveKit installation.
"""

import string


def is_filler_only(text: str, filler_words: set) -> bool:
    """
    Replicate the _is_filler_only logic from FillerFilteredAgentActivity.

    Args:
        text: Transcript text to check
        filler_words: Set of filler words

    Returns:
        True if text contains only filler words, False otherwise
    """
    if not text:
        return False

    # Normalize: lowercase, strip whitespace
    normalized = text.lower().strip()

    # Split into words and check if all are in filler set
    words = normalized.split()

    # Empty text after normalization
    if not words:
        return False

    # Strip punctuation from each word before checking
    # This handles cases like "uh!" or "umm." or "hmm,"
    cleaned_words = [word.strip(string.punctuation) for word in words]

    # Filter out empty strings after punctuation removal
    cleaned_words = [w for w in cleaned_words if w]

    if not cleaned_words:
        return False

    # Check if all words are in filler set
    return all(word in filler_words for word in cleaned_words)


def test_filler_detection():
    """Test the filler detection logic."""
    print("Testing Filler Detection Logic")
    print("=" * 60)

    # Default filler words
    filler_words = {"uh", "umm", "hmm", "haan", "huh"}

    # Test cases: (input, expected_is_filler, description)
    test_cases = [
        # Filler-only cases (should return True)
        ("uh", True, "Single filler word"),
        ("umm", True, "Single filler word"),
        ("hmm", True, "Single filler word"),
        ("huh", True, "Single filler word"),
        ("uh umm", True, "Multiple filler words"),
        ("hmm uh", True, "Multiple filler words"),
        ("UH", True, "Case insensitive"),
        ("Umm", True, "Case insensitive"),
        (" uh ", True, "With whitespace"),
        ("uh umm hmm", True, "Three filler words"),
        # Non-filler cases (should return False)
        ("wait", False, "Single non-filler word"),
        ("stop", False, "Single non-filler word"),
        ("wait one second", False, "Multiple non-filler words"),
        ("umm okay stop", False, "Mixed filler and non-filler"),
        ("uh but wait", False, "Mixed filler and non-filler"),
        ("hello", False, "Regular word"),
        ("", False, "Empty string"),
        ("   ", False, "Whitespace only"),
        ("okay umm", False, "Non-filler first"),
        # Punctuation cases (should handle gracefully)
        ("uh!", True, "Filler with exclamation"),
        ("umm.", True, "Filler with period"),
        ("hmm,", True, "Filler with comma"),
        ("uh? umm!", True, "Multiple fillers with punctuation"),
        ("uh, umm, hmm", True, "Fillers separated by commas"),
        ("wait!", False, "Non-filler with punctuation"),
        ("uh! stop", False, "Mixed with punctuation"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in test_cases:
        result = is_filler_only(text, filler_words)
        status = "✅" if result == expected else "❌"

        if result == expected:
            passed += 1
            print(f"{status} '{text}' → {result} | {description}")
        else:
            failed += 1
            print(f"{status} '{text}' → {result} (expected {expected}) | {description}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_scenarios():
    """Test all scenarios from task.md."""
    print("\n\nTesting Task.md Scenarios")
    print("=" * 60)

    filler_words = {"uh", "umm", "hmm", "haan", "huh"}

    scenarios = [
        {
            "num": 1,
            "agent_speaking": True,
            "input": "uh",
            "expected_behavior": "Ignore, continue speaking",
        },
        {
            "num": 2,
            "agent_speaking": True,
            "input": "hmm",
            "expected_behavior": "Ignore, continue speaking",
        },
        {
            "num": 3,
            "agent_speaking": True,
            "input": "umm",
            "expected_behavior": "Ignore, continue speaking",
        },
        {
            "num": 4,
            "agent_speaking": True,
            "input": "wait one second",
            "expected_behavior": "Immediate interruption",
        },
        {
            "num": 5,
            "agent_speaking": True,
            "input": "umm okay stop",
            "expected_behavior": "Immediate interruption",
        },
        {
            "num": 6,
            "agent_speaking": False,
            "input": "umm",
            "expected_behavior": "Register as valid speech",
        },
    ]

    all_passed = True

    for scenario in scenarios:
        print(f"\nScenario {scenario['num']}: '{scenario['input']}'")
        print(f"  Agent Speaking: {scenario['agent_speaking']}")
        print(f"  Expected: {scenario['expected_behavior']}")

        # Check if input is filler-only
        is_filler = is_filler_only(scenario["input"], filler_words)

        # Determine if should filter
        # Filter only if: agent is speaking AND input is filler-only
        should_filter = scenario["agent_speaking"] and is_filler

        # Verify expected behavior
        if scenario["expected_behavior"] == "Ignore, continue speaking":
            expected_filter = True
        else:
            expected_filter = False

        if should_filter == expected_filter:
            print(f"  Actual: {'Filtered (ignored)' if should_filter else 'Processed'}")
            print("  ✅ PASSED")
        else:
            print(f"  Actual: {'Filtered (ignored)' if should_filter else 'Processed'}")
            print(f"  ❌ FAILED (expected {'filter' if expected_filter else 'process'})")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All scenarios PASSED!")
    else:
        print("❌ Some scenarios FAILED")
    print("=" * 60)

    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n\nTesting Edge Cases")
    print("=" * 60)

    filler_words = {"uh", "umm", "hmm", "haan", "huh"}

    edge_cases = [
        ("uh huh", True, "Both words are fillers"),
        ("uh oh", False, "'oh' is not in filler list"),
        ("    uh    ", True, "Extra whitespace"),
        ("UH UMM HMM", True, "All uppercase"),
        ("Uh Umm Hmm", True, "Mixed case"),
        ("uh\numm", True, "With newline (becomes space after split)"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in edge_cases:
        result = is_filler_only(text, filler_words)
        status = "✅" if result == expected else "❌"

        if result == expected:
            passed += 1
            print(f"{status} '{repr(text)}' → {result} | {description}")
        else:
            failed += 1
            print(f"{status} '{repr(text)}' → {result} (expected {expected}) | {description}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Filler Filter Logic Verification")
    print("=" * 60)
    print()

    all_passed = True
    all_passed &= test_filler_detection()
    all_passed &= test_scenarios()
    all_passed &= test_edge_cases()

    print("\n\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe filler filter logic is correct and handles:")
        print("  • All 6 scenarios from task.md")
        print("  • Case-insensitive matching")
        print("  • Whitespace normalization")
        print("  • Mixed filler/non-filler detection")
        print("  • Edge cases")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
