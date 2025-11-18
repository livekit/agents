# examples/simulate_filler_simple.py
import asyncio
# relative import because this file lives inside the examples package
from .interrupt_handler_simple import SimpleInterruptionHandler

async def simulate_sequence():
    handler = SimpleInterruptionHandler()

    sequences = [
        (True, "umm"),
        (True, "umm stop"),
        (True, "hmm okay"),
        (False, "umm"),
        (False, "hello there"),
        (True, "uh no you can stop"),
    ]

    for speaking, text in sequences:
        handler.agent_is_speaking = speaking
        state = "AGENT SPEAKING" if speaking else "AGENT IDLE"
        print("\n---", state, "| input:", repr(text))
        await handler.handle_transcript(text, is_final=True)
        await asyncio.sleep(0.3)

if __name__ == "__main__":
    asyncio.run(simulate_sequence())
