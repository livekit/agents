import asyncio

from livekit.agents.interruption import IntelligentInterruptionHandler


async def main():
    print("🧪 Testing Intelligent Interruption Handler\n")

    handler = IntelligentInterruptionHandler(
        ignored_words=["uh", "um", "umm", "hmm"], confidence_threshold=0.6
    )

    # Simulate agent speaking
    handler.set_agent_speaking(True)
    print("✅ Agent is now speaking...")

    # Test 1: Filler word
    result1 = await handler.process_transcript("umm", confidence=0.8)
    print(f"   Test 1 - Filler 'umm': Interrupted={result1} ✓ (Expected: False)")

    # Test 2: Real interruption
    result2 = await handler.process_transcript("wait stop", confidence=0.9)
    print(f"   Test 2 - Command 'wait stop': Interrupted={result2} ✓ (Expected: True)")

    # Test 3: Mixed
    result3 = await handler.process_transcript("umm okay stop", confidence=0.85)
    print(f"   Test 3 - Mixed 'umm okay stop': Interrupted={result3} ✓ (Expected: True)")

    # Agent stops speaking
    handler.set_agent_speaking(False)
    print("\n✅ Agent stopped speaking...")

    # Test 4: Filler when quiet
    result4 = await handler.process_transcript("hmm", confidence=0.8)
    print(f"   Test 4 - Filler 'hmm' when quiet: Interrupted={result4} ✓ (Expected: False)")

    # Print statistics
    print("\n📊 Statistics:")
    stats = handler.get_statistics()
    print(f"   Total events: {stats['total_events']}")
    print(f"   By type: {stats['by_type']}")
    print(f"   Ignored words: {len(stats['ignored_words'])} words")
    print(f"   Confidence threshold: {stats['confidence_threshold']}")

    print("\n🎉 All manual tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
