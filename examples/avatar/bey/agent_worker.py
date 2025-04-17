import argparse
import logging
import sys
from functools import partial

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.plugins import bey, openai

logger = logging.getLogger("bey-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext, avatar_id: str | None):
    await ctx.connect()

    # initialize the local agent
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        # stt=deepgram.STT(),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=cartesia.TTS(),
    )

    bey_avatar = bey.AvatarSession(avatar_id=avatar_id)
    await bey_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        room_output_options=RoomOutputOptions(audio_enabled=False),  # forward audio to the avatar
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default avatar is Ege's Stock Avatar from https://docs.bey.dev/avatars/default
    parser.add_argument("--avatar-id", type=str)

    args, remaining_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_args  # pass remaining args to livekit cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_id=args.avatar_id),
            worker_type=WorkerType.ROOM,
        )
    )
