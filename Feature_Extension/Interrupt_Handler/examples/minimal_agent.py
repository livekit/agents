"""
Minimal demonstration of InterruptOrchestrator.
Runs a mock LiveKit agent session showing how to wire the callbacks.
"""

import asyncio
from ..src import IHConfig, SpeechGate, InterruptOrchestrator
from ..src.logkit import make_logger

log = make_logger("demo-agent")

class DummySession:
    """Mock AgentSession stand-in for testing locally without LiveKit Cloud."""
    def __init__(self):
        self.interrupted = False

    async def interrupt(self):
        self.interrupted = True
        log.info("DummySession.interrupt() called")

async def run_demo():
    cfg = IHConfig.from_env()
    gate = SpeechGate()
    sess = DummySession()
    orchestrator = InterruptOrchestrator(sess, gate, cfg)

    # simulate agent speaking
    gate.open()
    await orchestrator.on_transcription("umm hmm", confidence=0.9)
    await orchestrator.on_transcription("uh stop", confidence=0.9)
    # agent finished
    gate.close()
    await orchestrator.on_transcription("umm okay", confidence=0.9)
    print("\nInterrupted?", sess.interrupted)

if __name__ == "__main__":
    asyncio.run(run_demo())
