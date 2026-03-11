import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, spatialreal

logger = logging.getLogger("spatialreal-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    if not os.getenv("SPATIALREAL_API_KEY"):
        raise ValueError("SPATIALREAL_API_KEY is not set")

    if not os.getenv("SPATIALREAL_APP_ID"):
        raise ValueError("SPATIALREAL_APP_ID is not set")

    if not os.getenv("SPATIALREAL_AVATAR_ID"):
        raise ValueError("SPATIALREAL_AVATAR_ID is not set")

    spatialreal_avatar = spatialreal.AvatarSession()
    await spatialreal_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
