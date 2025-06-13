import logging
import os

from dotenv import load_dotenv
from PIL import Image

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import hedra, openai

logger = logging.getLogger("hedra-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    # upload an avatar image or use an avatar id from hedra
    avatar_image = Image.open(os.path.join(os.path.dirname(__file__), "avatar.jpg"))
    hedra_avatar = hedra.AvatarSession(avatar_image=avatar_image)
    await hedra_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
