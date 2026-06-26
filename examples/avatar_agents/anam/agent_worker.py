import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import anam, openai

logger = logging.getLogger("anam-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    anam_api_key = os.getenv("ANAM_API_KEY")
    if not anam_api_key:
        raise ValueError("ANAM_API_KEY is not set")

    anam_avatar_id = os.getenv("ANAM_AVATAR_ID")
    if not anam_avatar_id:
        raise ValueError("ANAM_AVATAR_ID is not set")

    anam_avatar = anam.AvatarSession(
        persona_config=anam.PersonaConfig(
            name="avatar",
            avatarId=anam_avatar_id,
        ),
        api_key=anam_api_key,
        # Optionally request explicit output dimensions (pixels). Omit to use the
        # avatar model's default. Supported pairs are model-dependent, e.g. Cara 4
        # landscape 1152x768.
        # session_options=anam.SessionOptions(video_width=1152, video_height=768),
    )
    await anam_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
