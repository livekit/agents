import asyncio
from interruption_handler.interrupt_handler import InterruptHandler, TranscriptEvent

async def simulate(handler):
    events = [
        (True, TranscriptEvent("uh", 0.95)),
        (True, TranscriptEvent("umm hmm", 0.90)),
        (True, TranscriptEvent("umm okay stop", 0.92)),
        (True, TranscriptEvent("hmm yeah", 0.20)),  # low confidence
        (False, TranscriptEvent("umm", 0.90)),
        (False, TranscriptEvent("hello, I need help", 0.95)),
    ]

    for speaking, ev in events:
        act, reason, text = await handler.on_transcript_event(speaking, ev)
        print(f"[agent_speaking={speaking}] '{text}' => {act} ({reason})")
        await asyncio.sleep(0.2)

async def main():
    handler = InterruptHandler()
    await simulate(handler)

if __name__ == "__main__":
    asyncio.run(main())
