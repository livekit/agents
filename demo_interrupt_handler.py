import asyncio
from agents.extensions.interrupt_handler import InterruptFilter

async def dummy_stop():
    print("🛑 Agent stopped speaking.")

async def main():
    # ✅ Pass 'None' as the session argument (required positional)
    handler = InterruptFilter(
        None,  # session placeholder
        stop_callback=dummy_stop,
        ignored_words=["uh", "umm", "hmm", "haan"],
    )

    print("✅ Interrupt handler initialized")

    # Simulate agent speaking
    await handler.set_agent_speaking(True)
    print("Agent is speaking...")

    # Case 1: User says filler
    user_text = "uh"
    if handler.is_filler_only(user_text):
        print(f"Ignored filler word: {user_text}")
    else:
        print(f"Processed user input: {user_text}")

    # Case 2: User interrupts with real command
    user_text = "stop please"
    if handler.is_filler_only(user_text):
        print(f"Ignored filler word: {user_text}")
    else:
        print(f"Detected real interruption: {user_text}")
        await dummy_stop()

    # Case 3: Agent is quiet
    await handler.set_agent_speaking(False)
    user_text = "umm"
    if handler.is_filler_only(user_text):
        print(f"🗣 Registered filler as valid speech (agent quiet): {user_text}")
    else:
        print(f"Processed input: {user_text}")

if __name__ == "__main__":
    asyncio.run(main())
