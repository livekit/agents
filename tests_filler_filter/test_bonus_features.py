"""
Test Suite for Bonus Features
- Dynamic filler updates via API
- Multi-language filler detection

Author: Raghav (LiveKit Intern Assessment)
Date: November 19, 2025
"""

import asyncio
import os
import sys

# Add the livekit-agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'livekit-agents'))

from livekit.agents.voice.filler_filter import FillerFilter


def test_multi_language_support():
    """Test multi-language filler detection and auto-switching."""

    print("=" * 80)
    print("TEST 1: MULTI-LANGUAGE SUPPORT")
    print("=" * 80)
    print()

    # Create filter with multi-language enabled
    filter_ml = FillerFilter(enable_multi_language=True, default_language="en")

    print("✓ Created multi-language filter (default: en)")
    print(f"  Available languages: {filter_ml.get_available_languages()}")
    print(f"  Current language: {filter_ml.get_current_language()}")
    print(f"  Current fillers: {filter_ml.get_ignored_words()}")
    print()

    test_cases = [
        {
            "name": "English filler (default language)",
            "text": "umm",
            "language": "en",
            "expected": True,
        },
        {
            "name": "Hindi filler (auto-switch)",
            "text": "haan",
            "language": "hi",
            "expected": True,
        },
        {
            "name": "Hindi non-filler",
            "text": "ruko",  # "wait" in Hindi
            "language": "hi",
            "expected": False,
        },
        {
            "name": "Spanish filler",
            "text": "eh pues",
            "language": "es",
            "expected": True,
        },
        {
            "name": "Spanish non-filler",
            "text": "espera",  # "wait" in Spanish
            "language": "es",
            "expected": False,
        },
    ]

    passed = 0
    for test in test_cases:
        result = filter_ml.is_filler_only(
            test["text"],
            confidence=0.8,
            agent_is_speaking=True,
            language=test["language"]
        )

        status = "✓ PASS" if result == test["expected"] else "✗ FAIL"
        emoji = "🟢" if result == test["expected"] else "🔴"

        print(f"{emoji} {status} - {test['name']}")
        print(f"   Input: '{test['text']}' (lang: {test['language']})")
        print(f"   Expected: {test['expected']}, Got: {result}")

        if result == test["expected"]:
            passed += 1

        # Show current state after auto-switch
        print(f"   Current language: {filter_ml.get_current_language()}")
        print()

    print("=" * 80)
    print(f"Multi-language tests: {passed}/{len(test_cases)} passed")
    print("=" * 80)
    print()

    return passed == len(test_cases)


def test_manual_language_switching():
    """Test manual language switching."""

    print("=" * 80)
    print("TEST 2: MANUAL LANGUAGE SWITCHING")
    print("=" * 80)
    print()

    filter_ml = FillerFilter(enable_multi_language=True)

    print("Test 1: Switch to Hindi")
    success = filter_ml.switch_language("hi")
    print(f"  Success: {success}")
    print(f"  Current language: {filter_ml.get_current_language()}")
    print(f"  Current fillers: {filter_ml.get_ignored_words()}")
    assert success and filter_ml.get_current_language() == "hi"
    print("  ✓ PASS")
    print()

    print("Test 2: Hindi filler should be detected")
    result = filter_ml.is_filler_only("haan arey", confidence=0.8)
    print(f"  'haan arey' is filler: {result}")
    assert result
    print("  ✓ PASS")
    print()

    print("Test 3: Switch to Spanish")
    success = filter_ml.switch_language("es")
    print(f"  Success: {success}")
    print(f"  Current language: {filter_ml.get_current_language()}")
    print(f"  Current fillers: {filter_ml.get_ignored_words()}")
    assert success and filter_ml.get_current_language() == "es"
    print("  ✓ PASS")
    print()

    print("Test 4: Spanish filler should be detected")
    result = filter_ml.is_filler_only("eh pues", confidence=0.8)
    print(f"  'eh pues' is filler: {result}")
    assert result
    print("  ✓ PASS")
    print()

    print("Test 5: Try to switch to unsupported language")
    success = filter_ml.switch_language("xyz")
    print(f"  Success: {success} (should be False)")
    assert not success
    print("  ✓ PASS")
    print()

    print("=" * 80)
    print("🎉 All manual language switching tests passed!")
    print("=" * 80)
    print()

    return True


def test_add_custom_language():
    """Test adding custom language fillers."""

    print("=" * 80)
    print("TEST 3: ADD CUSTOM LANGUAGE")
    print("=" * 80)
    print()

    filter_ml = FillerFilter(enable_multi_language=True)

    print("Test 1: Add custom language (Urdu)")
    urdu_fillers = ["achha", "theek", "haan", "ji"]
    filter_ml.add_language_fillers("ur", urdu_fillers)
    print(f"  Added Urdu fillers: {urdu_fillers}")
    print(f"  Available languages: {filter_ml.get_available_languages()}")
    assert "ur" in filter_ml.get_available_languages()
    print("  ✓ PASS")
    print()

    print("Test 2: Switch to Urdu")
    success = filter_ml.switch_language("ur")
    print(f"  Success: {success}")
    print(f"  Current fillers: {filter_ml.get_ignored_words()}")
    assert success and filter_ml.get_current_language() == "ur"
    print("  ✓ PASS")
    print()

    print("Test 3: Urdu filler should be detected")
    result = filter_ml.is_filler_only("achha ji", confidence=0.8)
    print(f"  'achha ji' is filler: {result}")
    assert result
    print("  ✓ PASS")
    print()

    print("=" * 80)
    print("🎉 All custom language tests passed!")
    print("=" * 80)
    print()

    return True


async def test_dynamic_updates():
    """Test dynamic runtime updates."""

    print("=" * 80)
    print("TEST 4: DYNAMIC RUNTIME UPDATES")
    print("=" * 80)
    print()

    filter_dynamic = FillerFilter()

    print(f"Initial fillers: {filter_dynamic.get_ignored_words()}")
    print()

    print("Test 1: Add new fillers dynamically")
    result = await filter_dynamic.update_fillers_dynamic(add=["arey", "yaar", "bas"])
    print(f"  Result: {result['status']}")
    print(f"  Added: {result['added']}")
    print(f"  Current fillers count: {len(result['current_fillers'])}")
    assert "arey" in result['current_fillers']
    assert "yaar" in result['current_fillers']
    print("  ✓ PASS")
    print()

    print("Test 2: Remove fillers dynamically")
    result = await filter_dynamic.update_fillers_dynamic(remove=["okay", "ok"])
    print(f"  Result: {result['status']}")
    print(f"  Removed: {result['removed']}")
    print(f"  Current fillers count: {len(result['current_fillers'])}")
    assert "okay" not in result['current_fillers']
    print("  ✓ PASS")
    print()

    print("Test 3: Add and remove simultaneously")
    result = await filter_dynamic.update_fillers_dynamic(
        add=["theek", "accha"],
        remove=["yeah", "yep"]
    )
    print(f"  Result: {result['status']}")
    print(f"  Added: {result['added']}")
    print(f"  Removed: {result['removed']}")
    print(f"  Current fillers: {result['current_fillers'][:5]}... ({len(result['current_fillers'])} total)")
    assert "theek" in result['current_fillers']
    assert "yeah" not in result['current_fillers']
    print("  ✓ PASS")
    print()

    print("Test 4: Verify updated fillers work")
    # "theek" was just added, should be filtered
    result_filler = filter_dynamic.is_filler_only("theek", confidence=0.8, agent_is_speaking=True)
    print(f"  'theek' is filler: {result_filler} (should be True)")
    assert result_filler

    # "yeah" was removed, should NOT be filtered
    result_not_filler = filter_dynamic.is_filler_only("yeah", confidence=0.8, agent_is_speaking=True)
    print(f"  'yeah' is filler: {result_not_filler} (should be False)")
    assert not result_not_filler
    print("  ✓ PASS")
    print()

    print("=" * 80)
    print("🎉 All dynamic update tests passed!")
    print("=" * 80)
    print()

    return True


def test_combined_features():
    """Test multi-language with dynamic updates."""

    print("=" * 80)
    print("TEST 5: COMBINED FEATURES (Multi-Lang + Dynamic)")
    print("=" * 80)
    print()

    filter_combined = FillerFilter(enable_multi_language=True, default_language="en")

    print("Test 1: Start with English")
    print(f"  Current language: {filter_combined.get_current_language()}")
    print(f"  Current fillers: {filter_combined.get_ignored_words()[:5]}...")
    print()

    print("Test 2: Auto-switch to Hindi via language parameter")
    result = filter_combined.is_filler_only(
        "haan arey",
        confidence=0.8,
        agent_is_speaking=True,
        language="hi"
    )
    print(f"  Detected 'haan arey' as filler: {result}")
    print(f"  Current language after auto-switch: {filter_combined.get_current_language()}")
    assert result
    assert filter_combined.get_current_language() == "hi"
    print("  ✓ PASS")
    print()

    print("Test 3: Dynamically add custom Hindi filler")
    asyncio.run(filter_combined.update_fillers_dynamic(add=["arre", "bhai"]))
    print("  Added custom Hindi fillers")
    print(f"  Current fillers: {filter_combined.get_ignored_words()}")
    print()

    print("Test 4: Verify custom filler works")
    result = filter_combined.is_filler_only("arre bhai", confidence=0.8)
    print(f"  'arre bhai' is filler: {result}")
    assert result
    print("  ✓ PASS")
    print()

    print("=" * 80)
    print("🎉 All combined feature tests passed!")
    print("=" * 80)
    print()

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "BONUS FEATURES TEST SUITE" + " " * 33 + "║")
    print("║" + " " * 15 + "Dynamic Updates + Multi-Language" + " " * 32 + "║")
    print("║" + " " * 30 + "by Raghav" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    all_passed = True

    # Test 1: Multi-language support
    if not test_multi_language_support():
        all_passed = False

    # Test 2: Manual language switching
    if not test_manual_language_switching():
        all_passed = False

    # Test 3: Add custom language
    if not test_add_custom_language():
        all_passed = False

    # Test 4: Dynamic updates
    if not asyncio.run(test_dynamic_updates()):
        all_passed = False

    # Test 5: Combined features
    if not test_combined_features():
        all_passed = False

    # Final summary
    print("\n")
    print("=" * 80)
    if all_passed:
        print("🎉🎉🎉 ALL BONUS FEATURE TESTS PASSED! 🎉🎉🎉")
        print()
        print("Bonus features implemented:")
        print("  ✓ Dynamic filler updates via async API")
        print("  ✓ Multi-language support (10 languages)")
        print("  ✓ Auto-language switching based on STT")
        print("  ✓ Manual language switching")
        print("  ✓ Custom language addition")
        print("  ✓ Runtime filler list updates")
    else:
        print("⚠️  SOME TESTS FAILED - Please review output above")
    print("=" * 80)
    print("\n")

    sys.exit(0 if all_passed else 1)
