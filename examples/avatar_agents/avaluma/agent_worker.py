import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import avaluma, openai

logger = logging.getLogger("avaluma-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    # AVALUMA_LICENSE_KEY and AVALUMA_AVATAR_ID can also be provided via environment variables
    avaluma_avatar = avaluma.AvatarSession(
        avatar_id=os.getenv("AVALUMA_AVATAR_ID"),
    )
    await avaluma_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
