import logging

from livekit.agents import JobContext, WorkerOptions, cli


async def entrypoint(ctx: JobContext):
    logging.info("starting entrypoint")

    # Connect to the room
    await ctx.connect()

    # Add your agent logic here!


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
