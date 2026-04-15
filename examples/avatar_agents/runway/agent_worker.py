import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import google, runway

logger = logging.getLogger("runway-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=google.realtime.RealtimeModel(voice="kore"),
        resume_false_interruption=False,
    )

    avatar_id = os.getenv("RUNWAY_AVATAR_ID")
    preset_id = os.getenv("RUNWAY_AVATAR_PRESET_ID")
    runway_avatar = runway.AvatarSession(avatar_id=avatar_id, preset_id=preset_id)
    await runway_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
