"""
Test script for FillerInterruptionHandler

This script tests the core logic of the filler handler without requiring
a full LiveKit setup. Useful for quick validation and debugging.

Run: python test_filler_handler.py
"""

import sys
from filler_interrupt_handler import FillerInterruptionHandler


def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print formatted test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"   └─ {details}")


def run_tests():
    """Run comprehensive tests on the filler handler."""
    print("=" * 60)
    print("Filler Interruption Handler - Test Suite")
    print("=" * 60)
    print()
    
    # Initialize handler with test filler words
    handler = FillerInterruptionHandler(
        ignored_words=['uh', 'um', 'umm', 'hmm', 'yeah', 'haan']
    )
    
    print(f"Handler initialized with {len(handler.ignored_words)} filler words")
    print(f"Filler words: {sorted(handler.ignored_words)}")
    print()
    print("-" * 60)
    print("Running Tests...")
    print("-" * 60)
    print()
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Filler during agent speech should be ignored
    total_tests += 1
    handler.update_agent_state('speaking')
    decision = handler.analyze_transcript('umm', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 1: Filler 'umm' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 2: Multiple fillers during agent speech
    total_tests += 1
    decision = handler.analyze_transcript('uh hmm yeah', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 2: Multiple fillers 'uh hmm yeah' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 3: Real speech during agent speaking should interrupt
    total_tests += 1
    decision = handler.analyze_transcript('wait a second', is_final=True)
    test_passed = decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 3: Real speech 'wait a second' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: True)"
    )
    
    # Test 4: Mixed filler and real speech should interrupt
    total_tests += 1
    decision = handler.analyze_transcript('umm okay stop', is_final=True)
    test_passed = decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 4: Mixed 'umm okay stop' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: True)"
    )
    
    # Test 5: Filler when agent NOT speaking should be processed
    total_tests += 1
    handler.update_agent_state('listening')
    decision = handler.analyze_transcript('umm', is_final=True)
    test_passed = decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 5: Filler 'umm' when agent is listening",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: True)"
    )
    
    # Test 6: Real speech when agent idle should be processed
    total_tests += 1
    handler.update_agent_state('idle')
    decision = handler.analyze_transcript('hello there', is_final=True)
    test_passed = decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 6: Real speech 'hello there' when agent idle",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: True)"
    )
    
    # Test 7: Empty transcript should be ignored
    total_tests += 1
    handler.update_agent_state('speaking')
    decision = handler.analyze_transcript('', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 7: Empty transcript during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 8: Punctuation handling
    total_tests += 1
    decision = handler.analyze_transcript('umm, hmm.', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 8: Fillers with punctuation 'umm, hmm.' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 9: Case insensitivity
    total_tests += 1
    decision = handler.analyze_transcript('UMM HMM', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 9: Case insensitive 'UMM HMM' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 10: Dynamic word addition
    total_tests += 1
    handler.add_ignored_words(['okay', 'alright'])
    decision = handler.analyze_transcript('okay alright', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 10: Dynamically added words 'okay alright' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    # Test 11: Dynamic word removal
    total_tests += 1
    handler.remove_ignored_words(['yeah'])
    decision = handler.analyze_transcript('yeah', is_final=True)
    test_passed = decision.should_interrupt  # Should now be treated as real speech
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 11: Removed word 'yeah' now treated as real speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: True)"
    )
    
    # Test 12: Multi-language support (Hindi example)
    total_tests += 1
    handler.add_ignored_words(['haan', 'achha', 'theek'])
    decision = handler.analyze_transcript('haan achha', is_final=True)
    test_passed = not decision.should_interrupt
    if test_passed:
        passed_tests += 1
    print_test_result(
        "Test 12: Hindi fillers 'haan achha' during agent speech",
        test_passed,
        f"Should interrupt: {decision.should_interrupt} (expected: False)"
    )
    
    print()
    print("-" * 60)
    print("Test Summary")
    print("-" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    # Print handler statistics
    print("-" * 60)
    print("Handler Statistics")
    print("-" * 60)
    handler.log_statistics()
    print()
    
    # Return exit code based on test results
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    print()
    exit_code = run_tests()
    
    if exit_code == 0:
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
    
    print()
    sys.exit(exit_code)
