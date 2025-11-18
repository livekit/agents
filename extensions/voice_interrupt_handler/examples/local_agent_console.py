import os, sys, asyncio, logging

# Ensure extension imports work
ext = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext not in sys.path:
    sys.path.insert(0, ext)

from livekit.agents import Agent, AgentSession
from livekit.plugins import silero
from voice_interrupt.plugin import attach_interrupt_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_console_agent")

async def main():
    agent = Agent(instructions="Local test agent with real mic input.")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=silero.STT(),
        llm=None,
        tts=silero.TTS(),
    )

    attach_interrupt_handler(
        session,
        ignored_words={"uh", "umm", "hmm", "haan"},
        stop_words={"stop", "wait"},
        min_confidence=0.0,
    )

    logger.info("Starting REAL local AgentSession with mic/speaker…")

    # Direct console audio mode WITHOUT Worker
    await session.console()   # <= IMPORTANT: uses local mic/speaker
    await session.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
