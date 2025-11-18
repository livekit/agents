# run_with_extension.py
"""
Run the extension. Attempts to start a real AgentSession; if Room/IO fails
(e.g. room=None), falls back to a deterministic mock simulation that demonstrates
all required behaviors (no external APIs, no audio devices required).
"""
import asyncio
import logging
import os
import sys

# Ensure extension package is importable when run directly
ext_path = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext_path not in sys.path:
    sys.path.insert(0, ext_path)

from voice_interrupt.plugin import attach_interrupt_handler

# Try to import real LiveKit classes; if not present, we'll still run the mock.
try:
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import silero
    LIVEKIT_AVAILABLE = True
except Exception:
    LIVEKIT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_with_extension")


async def run_mock_simulation():
    """
    Simulates agent state changes and ASR transcripts to demonstrate:
      - filler ignored while agent speaks
      - 'stop' interrupts agent immediately
      - filler accepted when agent is idle
    """
    from voice_interrupt.handler import InterruptHandler

    class MockSession:
        def __init__(self):
            self._callbacks = {}
            self.interrupted = 0
        def on(self, name, cb):
            logger.info(f"MOCK: registered callback -> {name}")
            self._callbacks[name] = cb
        def interrupt(self):
            self.interrupted += 1
            logger.info("MOCK: session.interrupt() called")

    mock = MockSession()
    handler = InterruptHandler(mock,
                               ignored_words={"uh", "umm", "hmm", "haan"},
                               stop_words={"stop", "wait"},
                               min_confidence=0.0)
    handler.start()

    # Use the same async handlers the real system would call
    logger.info("\n--- MOCK SIM: Agent starts speaking ---")
    await handler._on_agent_state_changed({"new_state": "speaking"})
    logger.info("MOCK SIM: filler 'uh' while agent speaking")
    await handler._on_user_input_transcribed({"is_final": True, "transcript": "uh", "confidence": 1.0})
    logger.info("MOCK SIM: meaningful 'stop' while agent speaking")
    await handler._on_user_input_transcribed({"is_final": True, "transcript": "stop", "confidence": 1.0})

    logger.info("\n--- MOCK SIM: Agent now idle ---")
    await handler._on_agent_state_changed({"new_state": "idle"})
    logger.info("MOCK SIM: filler 'umm' while agent idle")
    await handler._on_user_input_transcribed({"is_final": True, "transcript": "umm", "confidence": 1.0})

    logger.info(f"\nMOCK SIM DONE: session.interrupted = {mock.interrupted}")

    return True


async def main():
    # If LiveKit modules are available and you want to run a real session,
    # edit the block below to supply a real room or use the repo's worker/CLI.
    if LIVEKIT_AVAILABLE:
        try:
            # Create an Agent and an AgentSession with local plugins (silero)
            agent = Agent(instructions="You are a test assistant (local).")
            # NOTE: If your repo provides other plugin factories, replace these as needed.
            session = AgentSession(
                vad=silero.VAD.load(),
                stt=silero.STT(),   # local/offline STT (if available in your environment)
                tts=silero.TTS()    # local/offline TTS (if available)
            )

            # Attach your interrupt handler extension
            attach_interrupt_handler(session,
                                     ignored_words={"uh", "umm", "hmm", "haan"},
                                     stop_words={"stop", "wait"})

            # Try to start the session. If no valid room/IO is configured, this may raise.
            logger.info("Attempting to start real AgentSession (may require valid room/IO)...")
            await session.start(agent=agent, room=None)
            # If start succeeds, keep running
            await session.run_forever()
            return True

        except Exception as e:
            logger.warning("Real session start failed or is not configured (falling back to mock): %s", e)
            # fall through to run mock

    # If LIVEKIT not available or real start failed, run the mock simulation.
    await run_mock_simulation()
    return True


if __name__ == "__main__":
    asyncio.run(main())
