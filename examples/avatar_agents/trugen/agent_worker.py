import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import google, trugen

logger = logging.getLogger("trugen-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=google.realtime.RealtimeModel(voice="kore"),
        resume_false_interruption=False,
    )

    avatar_id = os.getenv("TRUGEN_AVATAR_ID")
    trugen_avatar = trugen.AvatarSession(avatar_id=avatar_id)
    await trugen_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="You are a friendly AI Agent."),
        room=ctx.room,
    )
    session.generate_reply(instructions="Greet the user.")


if __name__ == "__main__":
    cli.run_app(server)
