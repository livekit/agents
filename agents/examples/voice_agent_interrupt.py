import asyncio
from agents.extensions.interrupt_handler import InterruptHandler

# ANSI color codes for pretty console output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

async def simulate_livekit_stream():
    handler = InterruptHandler()
    total, ignored, interrupts = 0, 0, 0

    # ====== Simulate agent speaking ======
    print(f"\n{YELLOW}--- Simulating: Agent Speaking ---{RESET}")
    handler.set_agent_state(True)

    # Test phrases (English + Hinglish + edge cases)
    speaking_tests = [
        ("uh", 0.95),
        ("umm", 0.9),
        ("ummm", 0.92),
        ("hmmm okay", 0.9),
        ("haaan", 0.87),
        ("haan okay", 0.94),
        ("okay okay", 0.93),
        ("arre stop", 0.95),
        ("matlab wait", 0.96),
        ("umm stop", 0.93),
        ("hmm theek hai", 0.9),
        ("acha okay", 0.92),
        ("haan haan", 0.93),
        ("ha bhai", 0.9),
        ("arre okay", 0.9),
        ("so yeah stop", 0.95),
        ("ummm hmm", 0.88),
        ("haina", 0.92),
        ("okkk stop", 0.9),
        ("arre ruk jao", 0.94),
    ]

    for text, conf in speaking_tests:
        total += 1
        result = await handler.handle_transcript(text, conf)
        if result:
            print(f"{GREEN}✅ Detected real user interruption:{RESET} {result}")
            interrupts += 1
        else:
            ignored += 1
        await asyncio.sleep(0.2)

    # ====== Simulate agent silent period ======
    print(f"\n{YELLOW}--- Simulating: Agent Silent ---{RESET}")
    handler.set_agent_state(False)
    silent_tests = [
        ("umm", 0.9),
        ("haan okay", 0.95),
        ("accha theek hai", 0.93),
        ("stop please", 0.94),
        ("arre ek minute", 0.96),
        ("haan sahi", 0.9),
    ]

    for text, conf in silent_tests:
        total += 1
        result = await handler.handle_transcript(text, conf)
        if result:
            print(f"{GREEN}✅ Detected user speech:{RESET} {result}")
            interrupts += 1
        else:
            print(f"{RED}❌ Ignored while silent:{RESET} {text}")
            ignored += 1
        await asyncio.sleep(0.2)

    # ====== Summary ======
    print(f"\n{YELLOW}--- Test Summary ---{RESET}")
    print(f"Total phrases tested : {total}")
    print(f"Ignored fillers      : {ignored}")
    print(f"Detected interruptions: {interrupts}")

if __name__ == "__main__":
    asyncio.run(simulate_livekit_stream())
