# console_agent_minimal.py
"""
Minimal REAL audio console agent that works even when livekit.plugins is missing.
Uses AgentSession.console() which provides built-in mic/speaker IO.
No TTS/STT plugins needed. No LIVEKIT_URL needed.
"""

import os, sys, asyncio, logging

# ensure extension importable
ext = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext not in sys.path:
    sys.path.insert(0, ext)

from livekit.agents import Agent, AgentSession
from voice_interrupt.plugin import attach_interrupt_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("console_agent_minimal")

async def main():
    logger.info("Starting minimal local console agent…")

    # simple test agent
    agent = Agent(
        instructions="This is a minimal local console test agent. Speak to test interruptions."
    )

    # create session with no plugins
    session = AgentSession(
        vad=None,   # allow internal VAD
        stt=None,   # allow console STT
        llm=None,   # no language model
        tts=None    # allow console TTS
    )

    # attach your interrupt handler
    attach_interrupt_handler(
        session,
        ignored_words={"uh", "umm", "hmm", "haan"},
        stop_words={"stop", "wait"},
        min_confidence=0.0,
    )

    # start console audio mode
    logger.info("Calling session.console() — using local mic & speaker")
    await session.console()   # THIS enables real mic/speaker I/O

    # optional greeting speech
    try:
        await session.generate_reply(instructions="Hi, I am ready. Start speaking.")
    except:
        pass

    # keep agent running
    await session.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
