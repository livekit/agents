import sys
from livekit.agents.voice import FillerDetector

print("\n" + "=" * 70)
print(" " * 20 + "FILLER DETECTION TEST")
print("=" * 70 + "\n")

# Create detector
detector = FillerDetector(
    languages=['en', 'hi'],
    min_confidence_threshold=0.3,
    enable_logging=True,
)

print(f"✓ Detector initialized with:")
print(f"  - Languages: en, hi")
print(f"  - Filler words: {sorted(list(detector.filler_words))}")
print(f"  - Min confidence: 0.3")
print(f"  - Logging: enabled\n")

# Test cases with descriptions
test_cases = [
    ("uh", True, 0.9, "Pure filler while agent speaking"),
    ("umm okay stop", True, 0.9, "Mixed filler + command"),
    ("wait", True, 0.9, "Real interruption"),
    ("hmm", False, 0.9, "Filler when agent quiet"),
    ("hello", False, 0.9, "Real speech when quiet"),
    ("uh", True, 0.2, "Low confidence filler"),
    ("haan", True, 0.9, "Hindi filler word"),
    ("", True, 0.9, "Empty transcript"),
]

print("=" * 70)
print("TEST RESULTS")
print("=" * 70 + "\n")

for i, (text, agent_speaking, confidence, description) in enumerate(test_cases, 1):
    result = detector.detect(text, confidence, agent_speaking=agent_speaking)
    
    # Color coding for terminal (works in most terminals)
    status_icon = "✓" if not result.is_filler_only or not agent_speaking else "○"
    
    print(f"[Test {i}] {description}")
    print(f"  Input: '{text}'")
    print(f"  Context: Agent {'speaking' if agent_speaking else 'quiet'}, Confidence: {confidence}")
    print(f"  Result: {status_icon} {result.detection_reason}")
    print(f"  ├─ Is filler only: {result.is_filler_only}")
    print(f"  ├─ Should interrupt: {result.should_interrupt}")
    print(f"  └─ Filtered text: '{result.filtered_transcript}'")
    print()

print("=" * 70)
print("STATISTICS SUMMARY")
print("=" * 70 + "\n")

stats = detector.get_stats()
total = stats['total_transcripts']

print(f"Total transcripts processed: {total}")
print(f"├─ Filler-only (ignored):     {stats['filler_only_ignored']} ({stats['filler_only_ignored']/total*100:.1f}%)")
print(f"├─ Meaningful interruptions:  {stats['meaningful_interruptions']} ({stats['meaningful_interruptions']/total*100:.1f}%)")
print(f"├─ Low confidence (ignored):  {stats['low_confidence_ignored']} ({stats['low_confidence_ignored']/total*100:.1f}%)")
print(f"├─ Agent quiet (valid):       {stats['agent_quiet_valid']} ({stats['agent_quiet_valid']/total*100:.1f}%)")
print(f"└─ Empty transcripts:         {stats['empty_transcripts']} ({stats['empty_transcripts']/total*100:.1f}%)")

print("\n" + "=" * 70)
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 70 + "\n")

sys.exit(0)
