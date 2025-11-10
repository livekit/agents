"""
agent_with_filter_demo.py

Run: python -m examples.agent_with_filter_demo
(or: python examples\agent_with_filter_demo.py when run from repo root)

This demo shows InterruptFilter used as a single shared instance that
would be created at agent startup in a real agent.
"""

import asyncio
import os
import logging

from src.livekit_interrupt_filter import InterruptFilter

logging.basicConfig(level=logging.INFO)

# set example ignored words (optional)
os.environ["IGNORED_WORDS"] = "uh,umm,hmm,haan,mm,mmm"

# --- shared InterruptFilter instance (module-level) ---
interrupt_filter = InterruptFilter(
    ignored_words=["uh", "umm", "hmm", "haan"],
    filler_confidence_threshold=0.8,
    ignore_when_confidence_less_than=0.5,
)
# reload from env if available (runs as a background task if loop is running)
try:
    asyncio.get_event_loop().create_task(interrupt_filter.reload_from_env())
except RuntimeError:
    # event loop not running at import time; reload in startup if needed
    pass
# -------------------------------------------------------

async def mock_event_producer(agent):
    # transcripts: (text, confidence, is_final, delay)
    transcripts = [
        ("uh", 0.95, True, 0.2),
        ("umm okay stop", 0.9, True, 0.6),
        ("umm", 0.92, True, 0.1),
        ("wait one second", 0.98, True, 0.3),
        ("hmm yeah", 0.4, True, 0.2),
        ("I need to change that", 0.93, True, 0.4),
    ]
    for text, conf, is_final, delay in transcripts:
        await asyncio.sleep(delay)
        print("\n== ASR EVENT: text=%r conf=%.2f agent_speaking=%s ==" % (text, conf, agent.is_speaking))
        # use the shared module-level filter
        decision = await interrupt_filter.handle_transcription_event(text, conf, is_final, agent.is_speaking)
        print("DECISION:", decision)

        if decision["action"] == "stop":
            print(">>> ACTION: Stop agent TTS (simulate).")
            agent.is_speaking = False
        elif decision["action"] == "ignore":
            print(">>> ACTION: Ignore - do nothing.")
        elif decision["action"] == "register":
            print(">>> ACTION: Register speech for processing (send to agent).")

class MockAgent:
    def __init__(self):
        self.is_speaking = True

async def demo_main():
    agent = MockAgent()
    await mock_event_producer(agent)
    print("\nStats:", interrupt_filter.stats())

if __name__ == "__main__":
    # Run as a module or script:
    # from repo root: python -m examples.agent_with_filter_demo
    asyncio.run(demo_main())



