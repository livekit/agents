import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import kfl

logger = logging.getLogger("kfl-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4o"),
        tts=inference.TTS("cartesia/sonic-2"),
        resume_false_interruption=False,
    )

    kfl_persona_id = os.getenv("KFL_PERSONA_ID")
    kfl_persona_slug = os.getenv("KFL_PERSONA_SLUG")

    avatar = kfl.AvatarSession(
        persona_id=kfl_persona_id,
        persona_slug=kfl_persona_slug,
    )
    await avatar.start(session, room=ctx.room)

    agent = Agent(instructions="Talk to me!")

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
