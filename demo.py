"""
Demo script to showcase the interruption handler functionality.
"""

from interruption_handler import InterruptionHandler, InterruptionConfig


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_scenario(handler, text, confidence=None, description=""):
    """Test a single scenario and print the result."""
    should_ignore = handler.should_ignore_speech(text, confidence)
    
    confidence_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
    status = "🔇 IGNORED" if should_ignore else "✅ PROCESSED"
    
    print(f"{status} | '{text}'{confidence_str}")
    if description:
        print(f"         └─ {description}")


def main():
    """Run the demo."""
    print("\n" + "🎙️ " * 20)
    print_header("LiveKit Voice Interruption Handler Demo")
    print("🎙️ " * 20)
    
    # Create handler with default configuration
    config = InterruptionConfig.from_word_list(
        words=["uh", "umm", "hmm", "haan", "um", "er", "ah"],
        confidence_threshold=0.5
    )
    handler = InterruptionHandler(config)
    
    print(f"\n📋 Configuration:")
    print(f"   - Ignored words: {sorted(handler.get_ignored_words())}")
    print(f"   - Confidence threshold: {config.confidence_threshold}")
    
    # Scenario 1: Filler words only
    print_header("Scenario 1: Filler Words Only")
    print("Agent is speaking, user makes filler sounds...")
    print()
    
    test_scenario(handler, "uh", description="Single filler")
    test_scenario(handler, "umm", description="Another filler")
    test_scenario(handler, "uh umm hmm", description="Multiple fillers")
    test_scenario(handler, "Uh, umm... hmm", description="Fillers with punctuation")
    
    # Scenario 2: Real interruptions
    print_header("Scenario 2: Real Interruptions")
    print("Agent is speaking, user wants to interrupt...")
    print()
    
    test_scenario(handler, "wait", description="Clear command")
    test_scenario(handler, "stop", description="Stop command")
    test_scenario(handler, "hold on", description="Multi-word command")
    test_scenario(handler, "I have a question", description="Full sentence")
    
    # Scenario 3: Mixed speech
    print_header("Scenario 3: Mixed Speech (Fillers + Real Words)")
    print("Agent is speaking, user hesitates but has real intent...")
    print()
    
    test_scenario(handler, "umm okay stop", description="Filler + command")
    test_scenario(handler, "uh wait a minute", description="Filler + request")
    test_scenario(handler, "hmm I think", description="Filler + thought")
    
    # Scenario 4: Confidence-based filtering
    print_header("Scenario 4: Confidence-Based Filtering")
    print("Testing low vs high confidence speech...")
    print()
    
    test_scenario(handler, "hello", confidence=0.2, description="Very low confidence")
    test_scenario(handler, "hello", confidence=0.4, description="Low confidence")
    test_scenario(handler, "hello", confidence=0.6, description="High confidence")
    test_scenario(handler, "hello", confidence=0.9, description="Very high confidence")
    
    # Scenario 5: Background noise
    print_header("Scenario 5: Background Noise & Murmurs")
    print("Testing background sounds that shouldn't interrupt...")
    print()
    
    test_scenario(handler, "hmm yeah", confidence=0.3, description="Low confidence murmur")
    test_scenario(handler, "uh huh", confidence=0.4, description="Acknowledgment sound")
    test_scenario(handler, "", description="Empty/silence")
    test_scenario(handler, "...", description="Punctuation only")
    
    # Scenario 6: Multilingual
    print_header("Scenario 6: Multilingual Fillers")
    print("Testing fillers from different languages...")
    print()
    
    test_scenario(handler, "haan", description="Hindi filler")
    test_scenario(handler, "uh haan", description="Mixed language fillers")
    test_scenario(handler, "namaste", description="Real Hindi word")
    
    # Scenario 7: Edge cases
    print_header("Scenario 7: Edge Cases")
    print("Testing unusual inputs...")
    print()
    
    test_scenario(handler, "UH", description="Uppercase filler")
    test_scenario(handler, "Umm", description="Mixed case filler")
    test_scenario(handler, "   ", description="Whitespace only")
    test_scenario(handler, "uh!", description="Filler with exclamation")
    
    # Summary
    print_header("Summary")
    stats = handler.get_stats()
    print(f"\n📊 Handler Statistics:")
    print(f"   - Total ignored words: {stats['ignored_words_count']}")
    print(f"   - Confidence threshold: {stats['confidence_threshold']}")
    print(f"   - Dynamic updates enabled: {stats['dynamic_updates_enabled']}")
    
    print("\n✨ Demo completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   1. Filler-only speech is ignored → Agent continues speaking")
    print("   2. Real words trigger interruption → Agent stops")
    print("   3. Low confidence speech is filtered → Prevents false positives")
    print("   4. Mixed speech (fillers + real words) → Treated as valid")
    print("   5. Case-insensitive and punctuation-tolerant")
    
    print("\n" + "🎙️ " * 20 + "\n")


if __name__ == "__main__":
    main()

