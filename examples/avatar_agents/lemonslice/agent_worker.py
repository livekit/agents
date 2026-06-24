import json
import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import lemonslice

logger = logging.getLogger("lemonslice-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": False,
            },
        ),
    )

    lemonslice_image_url = os.getenv("LEMONSLICE_IMAGE_URL")
    if lemonslice_image_url is None:
        raise ValueError("LEMONSLICE_IMAGE_URL must be set")
    avatar = lemonslice.AvatarSession(
        agent_image_url=lemonslice_image_url,
        # Prompt to guide the avatar's movements
        agent_prompt="Be expressive in your movements and use your hands while talking.",
    )
    await avatar.start(session, room=ctx.room)

    # Provide `meeting_url` as job metadatafor the avatar to join a 3rd party meeting platform (Zoom, Meet, Teams)
    # If not provided, the avatar will run as a standard LiveKit room avatar
    meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    meeting_url = meta.get("meeting_url")
    bot_name = meta.get("bot_name")
    if meeting_url:
        join_kwargs: dict[str, str] = {}
        if bot_name:
            join_kwargs["bot_name"] = bot_name
        await avatar.join_meeting(meeting_url, **join_kwargs)
        room_options = avatar.room_options()
    else:
        room_options = room_io.RoomOptions()

    agent = Agent(instructions="Talk to me!")

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_options,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
