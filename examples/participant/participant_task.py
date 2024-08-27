import asyncio
import logging

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli
from dataclasses import dataclass

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("connected to the room")

    async def participant_task(p: rtc.RemoteParticipant):
        logger.info(f"participant task starting for {p.identity}")
        # Do something with p.attributes, p.identity, p.metadata, etc.
        # my_stuff = await fetch_stuff_from_my_db(p)

        # Do something
        await asyncio.sleep(60)
        logger.info(f"participant task done for {p.identity}")

    await ctx.add_participant_task(task_fnc=participant_task, filter_fnc=lambda p: True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
