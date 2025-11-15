import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, heygen, openai

logger = logging.getLogger("heygen-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        resume_false_interruption=False,
    )

    heygen_avatar_id = os.getenv("HEYGEN_AVATAR_ID")
    avatar = heygen.AvatarSession(avatar_id=heygen_avatar_id)

    agent = Agent(instructions="Talk to me!")

    await session.start(
        agent=agent,
        room=ctx.room,
    )
    await avatar.start(session, room=ctx.room)

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
