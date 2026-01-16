import sys
import os

# Read and execute the interruption_filter.py file directly
filter_path = os.path.join('livekit-agents', 'livekit', 'agents', 'voice', 'interruption_filter.py')

with open(filter_path, 'r') as f:
    code = f.read()

# Create a namespace for the module
namespace = {}
exec(code, namespace)

# Get the InterruptionFilter class
InterruptionFilter = namespace['InterruptionFilter']

print("="*60)
print("LiveKit Interruption Filter - Direct Test")
print("="*60)

# Create filter
f = InterruptionFilter()
print(f"\n Filter created successfully")
print(f"  - Enabled: {f.enabled}")
print(f"  - Ignore words count: {len(f.ignore_words)}")
print(f"  - Sample ignore words: {list(f.ignore_words)[:5]}")

# Test 1: Backchanneling while speaking (should be ignored)
result1 = f.should_ignore_interruption("yeah", True)
print(f"\n Test 1 - Backchanneling while speaking")
print(f"  Input: 'yeah', Agent speaking: True")
print(f"  Should ignore: {result1} (expected: True)")
assert result1 == True, "FAILED: Should ignore 'yeah' when agent is speaking"

# Test 2: Backchanneling while silent (should NOT be ignored)
result2 = f.should_ignore_interruption("yeah", False)
print(f"\n Test 2 - Backchanneling while silent")
print(f"  Input: 'yeah', Agent speaking: False")
print(f"  Should ignore: {result2} (expected: False)")
assert result2 == False, "FAILED: Should NOT ignore 'yeah' when agent is silent"

# Test 3: Real interruption (should NOT be ignored)
result3 = f.should_ignore_interruption("stop", True)
print(f"\n Test 3 - Real interruption")
print(f"  Input: 'stop', Agent speaking: True")
print(f"  Should ignore: {result3} (expected: False)")
assert result3 == False, "FAILED: Should NOT ignore 'stop' even when agent is speaking"

# Test 4: Mixed input (should NOT be ignored)
result4 = f.should_ignore_interruption("yeah wait", True)
print(f"\n Test 4 - Mixed input")
print(f"  Input: 'yeah wait', Agent speaking: True")
print(f"  Should ignore: {result4} (expected: False)")
assert result4 == False, "FAILED: Should NOT ignore 'yeah wait' (contains command)"

# Test 5: Multiple backchanneling words
result5 = f.should_ignore_interruption("yeah okay hmm", True)
print(f"\n Test 5 - Multiple backchanneling words")
print(f"  Input: 'yeah okay hmm', Agent speaking: True")
print(f"  Should ignore: {result5} (expected: True)")
assert result5 == True, "FAILED: Should ignore multiple backchanneling words"

# Test 6: Case insensitivity
result6 = f.should_ignore_interruption("YEAH", True)
print(f"\n Test 6 - Case insensitivity")
print(f"  Input: 'YEAH', Agent speaking: True")
print(f"  Should ignore: {result6} (expected: True)")
assert result6 == True, "FAILED: Should be case-insensitive by default"

print("\n" + "="*60)
print(" All 6 tests passed! The interruption filter is working correctly.")
print("="*60)

print("\n" + "Assignment Test Scenarios:")
print("-"*60)

# Scenario 1: Long explanation
print("\n Scenario 1: Long Explanation")
print("  Agent speaking, user says 'okay yeah uh-huh'")
assert f.should_ignore_interruption("okay yeah uh-huh", True) == True
print("  Result: PASS - Agent continues speaking")

# Scenario 2: Passive affirmation
print("\n Scenario 2: Passive Affirmation")
print("  Agent silent, user says 'yeah'")
assert f.should_ignore_interruption("yeah", False) == False
print("  Result: PASS - Agent processes as answer")

# Scenario 3: The correction
print("\n Scenario 3: The Correction")
print("  Agent speaking, user says 'no stop'")
assert f.should_ignore_interruption("no stop", True) == False
print("  Result: PASS - Agent stops immediately")

# Scenario 4: Mixed input
print("\n Scenario 4: Mixed Input")
print("  Agent speaking, user says 'yeah okay but wait'")
assert f.should_ignore_interruption("yeah okay but wait", True) == False
print("  Result: PASS - Agent stops (contains 'wait')")

print("\n" + "="*60)
print(" All assignment scenarios passed!")
print("="*60)
