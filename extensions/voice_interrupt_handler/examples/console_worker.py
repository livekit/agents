# console_worker.py
"""
Run a local console-mode Agent that uses the extension without touching SDK or repo root.
Place this file under: extensions/voice_interrupt_handler/examples/console_worker.py
Run: python console_worker.py console
"""

import os
import sys
import logging

# make extension importable from here
ext_dir = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# Import Agents CLI and plugins (no SDK change)
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero
from voice_interrupt.plugin import attach_interrupt_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("console_worker")

@function_tool
async def dummy_tool(context: RunContext):
    return {"ok": True}

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = Agent(instructions="Local test agent for interrupt-handler demo.", tools=[dummy_tool])

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=silero.STT(),   # local/offline STT if available
        llm=None,          # keep local; not required for interruption demo
        tts=silero.TTS(),  # local TTS to hear agent (if available)
    )

    # Attach your extension (no SDK edits)
    attach_interrupt_handler(session,
                             ignored_words={"uh","umm","hmm","haan"},
                             stop_words={"stop","wait"},
                             min_confidence=0.0)

    # Start session in console mode (ctx.room provided by CLI runner)
    logger.info("Starting AgentSession in console mode (local mic/speaker).")
    await session.start(agent=agent, room=ctx.room)

    # Optionally prompt agent to speak once to begin demonstration
    try:
        await session.generate_reply(instructions="Hello — this is a local test. Please try saying uh, umm, and stop while I speak.")
    except Exception:
        pass

    await session.run_forever()

if __name__ == "__main__":
    # run via: python console_worker.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
