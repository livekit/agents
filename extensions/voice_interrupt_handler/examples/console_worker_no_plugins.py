# console_worker_no_plugins.py
"""
Run the Agent via the Agents CLI in console mode.
This avoids calling session.console() (not available), and avoids importing livekit.plugins.
The script lives inside the extension folder (no SDK edits).
Run:
    python .\extensions\voice_interrupt_handler\examples\console_worker_no_plugins.py console
"""

import os, sys, logging, asyncio
# make extension importable
ext = os.path.join(os.getcwd(), "extensions", "voice_interrupt_handler")
if ext not in sys.path:
    sys.path.insert(0, ext)

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from voice_interrupt.plugin import attach_interrupt_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("console_worker_no_plugins")

@function_tool
async def dummy_tool(ctx, *args, **kwargs):
    return {"ok": True}

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = Agent(instructions="Local console agent (no plugin imports).", tools=[dummy_tool])

    # Create session without importing local plugin modules
    session = AgentSession(
        vad=None,
        stt=None,
        llm=None,
        tts=None,
    )

    attach_interrupt_handler(
        session,
        ignored_words={"uh","umm","hmm","haan"},
        stop_words={"stop","wait"},
        min_confidence=0.0,
    )

    logger.info("Starting session via CLI-provided console room/IO (ctx.room).")
    await session.start(agent=agent, room=ctx.room)

    # Optional initial reply: only call if an LLM is configured
    try:
        if getattr(session, "llm", None) is not None:
            await session.generate_reply(instructions="Hello — ready to test interruptions.")
        else:
            logger.info("No LLM configured; skipping generate_reply().")
    except Exception:
        logger.exception("generate_reply non-fatal")

    # Keep the job alive. Some agent versions don't expose run_forever(); use a waiting Event.
    logger.info("Session started. Entering idle wait loop — use Ctrl+C to stop.")
    wait_evt = asyncio.Event()
    try:
        await wait_evt.wait()  # wait forever until process is terminated
    except asyncio.CancelledError:
        logger.info("Job cancelled; shutting down.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
