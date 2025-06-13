import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import xai

logger = logging.getLogger("xai-realtime-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=xai.RealtimeModel(),
    )

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
