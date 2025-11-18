"""
Simple simulator to test the InterruptHandler without LiveKit.
"""

import asyncio
from interrupt_handler import InterruptHandler

async def main():
    ih = InterruptHandler(confidence_threshold=0.4)

    print("\n--- Simulation Start ---\n")

    # agent starts speaking
    await ih.on_agent_state_change(True)

    # Test case 1: Agent speaking, user says filler word
    print("Agent speaking, user says 'uh' (filler)")
    result = await ih.on_transcription_event("uh", confidence=0.95)
    print(f"User said: 'uh' → Handler Result: {result}\n")

    # Test case 2: Agent speaking, user says another filler word
    print("Agent speaking, user says 'umm' (filler)")
    result = await ih.on_transcription_event("umm", confidence=0.9)
    print(f"User said: 'umm' → Handler Result: {result}\n")

    # Test case 3: Agent speaking, user says a stop command
    print("Agent speaking, user says 'umm okay stop' (stop command)")
    result = await ih.on_transcription_event("umm okay stop", confidence=0.96)
    print(f"User said: 'umm okay stop' → Handler Result: {result}\n")

    # Test case 4: Agent speaking, user says something with low confidence
    print("Agent speaking, user says 'background noise' (low confidence)")
    result = await ih.on_transcription_event("background noise", confidence=0.2)
    print(f"User said: 'background noise' → Handler Result: {result}\n")

    # agent stops speaking
    await ih.on_agent_state_change(False)
    # Test case 5: Agent not speaking, user says filler word (should be user_speech)
    print("Agent not speaking, user says 'umm' (should be user_speech)")
    result = await ih.on_transcription_event("umm", confidence=0.9)
    print(f"User said: 'umm' → Handler Result: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())
