import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli

logger = logging.getLogger("minimal-worker")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()  # connect to the RTC room

    logger.info(f"connected to the room {ctx.room.name}")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
