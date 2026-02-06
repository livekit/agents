import logging

from dotenv import load_dotenv

from livekit.agents import AgentServer, JobContext, cli

logger = logging.getLogger("minimal-worker")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"connected to the room {ctx.room.name}")


if __name__ == "__main__":
    cli.run_app(server)
