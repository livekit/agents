import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import liveavatar

logger = logging.getLogger("liveavatar-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        resume_false_interruption=False,
    )

    # Default to a public male avatar if not specified
    liveavatar_avatar_id = os.getenv("LIVEAVATAR_AVATAR_ID", "7b888024-f8c9-4205-95e1-78ce01497bda")
    avatar = liveavatar.AvatarSession(avatar_id=liveavatar_avatar_id)
    await avatar.start(session, room=ctx.room)

    agent = Agent(instructions="Talk to me!")

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
