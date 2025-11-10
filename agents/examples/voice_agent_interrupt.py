

import asyncio
from agents.extensions.interrupt_handler import InterruptHandler

async def simulate_livekit_stream():
    handler = InterruptHandler()

    # Simulate agent speaking
    handler.set_agent_state(True)

    # Simulated ASR events (transcripts + confidence)
    transcripts = [
        ("uh", 0.9),
        ("umm", 0.95),
        ("stop", 0.92),
        ("hmm okay stop", 0.85),
        ("haan", 0.88),
        ("wait one second", 0.95),
    ]

    for text, conf in transcripts:
        result = await handler.handle_transcript(text, conf)
        if result:
            print(f"✅ Detected real user interruption: {result}")
            # In real scenario → trigger agent.stop_speaking()
        await asyncio.sleep(0.5)

    # Simulate agent silent period
    handler.set_agent_state(False)
    await handler.handle_transcript("umm", 0.9)
    await handler.handle_transcript("haan okay", 0.95)

if __name__ == "__main__":
    asyncio.run(simulate_livekit_stream())


