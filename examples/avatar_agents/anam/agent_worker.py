import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import openai
from livekit.plugins import anam

logger = logging.getLogger("anam-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    anam_api_key = os.getenv("ANAM_API_KEY")
    if not anam_api_key:
        raise ValueError("ANAM_API_KEY is not set")

    avatar_id = os.getenv("ANAM_AVATAR_ID", "30fa96d0-26c4-4e55-94a0-517025942e18")

    anam_avatar = anam.AvatarSession(
        persona_config={
            "avatarId": avatar_id,
        },
        api_key=anam_api_key,
    )
    await anam_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
