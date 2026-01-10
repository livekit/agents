import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import avatario, openai

logger = logging.getLogger("avatario-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    avatar_id = os.getenv("AVATARIO_AVATAR_ID")
    avatario_avatar = avatario.AvatarSession(
        avatar_id=avatar_id,
        video_info=avatario.AvatarSession.VideoInfo(
            custom_background_url="https://images.pexels.com/photos/259915/pexels-photo-259915.jpeg",
            video_height=720,
            video_width=1280,
        ),
    )
    await avatario_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
