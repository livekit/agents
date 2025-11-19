import logging
import os

from dotenv import load_dotenv
from PIL import Image

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    room_io,
)
from livekit.plugins import bithuman, openai

logger = logging.getLogger("bithuman-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    bithuman_avatar = bithuman.AvatarSession(
        avatar_image=Image.open(os.path.join(os.path.dirname(__file__), "avatar.jpg")),
        # avatar_id="A70MXC2074",
        # model="expression",
    )
    await bithuman_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        # audio is forwarded to the avatar, so we disable room audio output
        room_options=room_io.RoomOptions(audio_output=False),
    )


if __name__ == "__main__":
    cli.run_app(server)
