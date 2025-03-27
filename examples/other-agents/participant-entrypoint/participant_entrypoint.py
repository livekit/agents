import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    async def participant_task_1(ctx: JobContext, p: rtc.RemoteParticipant):
        # You can filter out participants you are not interested in
        # if p.identity != "some_identity_of_interest":
        # return

        logger.info(f"participant task 1 starting for {p.identity}")
        # Do something with p.attributes, p.identity, p.metadata, etc.
        # my_stuff = await fetch_stuff_from_my_db(p)

        # Do something
        await asyncio.sleep(60)
        logger.info(f"participant task done for {p.identity}")

    async def participant_task_2(ctx: JobContext, p: rtc.RemoteParticipant):
        # multiple tasks can be run concurrently for each participant
        logger.info(f"participant task 2 starting for {p.identity}")
        await asyncio.sleep(10)

    # Add participant entrypoints before calling ctx.connect
    ctx.add_participant_entrypoint(entrypoint_fnc=participant_task_1)
    ctx.add_participant_entrypoint(entrypoint_fnc=participant_task_2)

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("connected to the room")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
