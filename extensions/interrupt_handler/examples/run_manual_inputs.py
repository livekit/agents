import asyncio
from livekit_interrupt_handler import InterruptHandler

async def main():
    handler = InterruptHandler()

    print("\n=== Manual Interrupt Handler Test ===")
    print("Agent will start speaking now.\n")
    await handler.set_agent_speaking(True)

    samples = [
        # --- Pure fillers ---
        ("uh", 0.95),
        ("umm", 0.92),
        ("hmm", 0.94),
        ("haan", 0.95),
        ("ehh", 0.91),
        ("mmm", 0.93),
        ("uhhh", 0.92),
        ("er", 0.90),

        # --- Mixed filler + command ---
        ("umm actually wait", 0.90),
        ("uh okay stop", 0.92),
        ("umm no no hold on", 0.95),
        ("eh wait wait", 0.91),
        ("haan please stop", 0.90),

        # --- Pure commands ---
        ("stop", 0.95),
        ("wait", 0.92),
        ("pause", 0.93),
        ("hold on", 0.95),
        ("no stop stop", 0.94),
        ("please wait", 0.95),
        ("just a second", 0.92),

        # --- Background noise / garbage ---
        ("unintelligible", 0.30),
        ("background talking", 0.35),
        ("random noise words", 0.25),
        ("asfdasf asdf asdf", 0.20),
        ("someone else talking", 0.30),

        # --- Long interruptions ---
        ("hey sorry to interrupt but I think you misunderstood the part about the billing cycle, can you clarify?", 0.85),
        ("wait I have another doubt regarding the second step you explained because I'm still confused", 0.88),

        # --- Filler-led long interruptions ---
        ("um yeah actually can you stop and listen to what I'm trying to say?", 0.89),

        # --- Low confidence ---
        ("uh okay fine", 0.29),

        # --- Multilingual filler/commands ---
        ("haan okay wait", 0.92),
        ("arre ruk jao", 0.94),
        ("eh un momento", 0.89),
        ("uh bitte warten", 0.91),

        # --- Command hidden in sentence ---
        ("sorry uh could you stop speaking so I can respond properly", 0.93),

        # --- Polite non-interruptions ---
        ("please continue", 0.95),
        ("no keep going", 0.95),

        # --- Negation-based interruption ---
        ("no wait actually", 0.95),

        # --- Sarcastic stretched tokens ---
        ("ummmmmmmm maybe stopppppp", 0.90),

        # --- Rapid bursts ---
        ("wait wait wait", 0.95),
        ("no no no no", 0.95),
    ]

    for text, conf in samples:
        res = await handler.on_transcription({"text": text, "confidence": conf})
        print(f"User said (agent speaking): '{text}' (conf={conf}) -> {res}")

    print("\nAgent finishes speaking now.\n")
    await handler.set_agent_speaking(False)

    # Agent silent tests:
    samples2 = [
        ("umm", 0.95),
        ("hello", 0.90),
        ("okay", 0.80),
        ("hey so basically I wanted to ask about the pricing tiers", 0.90),
        ("umm yeah okay let's continue", 0.95),
        ("please continue", 0.95),
    ]

    for text, conf in samples2:
        res = await handler.on_transcription({"text": text, "confidence": conf})
        print(f"User said (agent silent): '{text}' (conf={conf}) -> {res}")


if __name__ == "__main__":
    asyncio.run(main())
