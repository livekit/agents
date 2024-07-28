import logging

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    logger.info("connected to the room")
    # add your agent logic here!


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
