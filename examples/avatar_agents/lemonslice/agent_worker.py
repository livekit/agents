import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import lemonslice

logger = logging.getLogger("lemonslice-avatar-example")
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

    lemonslice_image_url = os.getenv("LEMONSLICE_IMAGE_URL")
    if lemonslice_image_url is None:
        raise ValueError("LEMONSLICE_IMAGE_URL must be set")
    avatar = lemonslice.AvatarSession(
        agent_image_url=lemonslice_image_url,
        # Prompt to guide the avatar's movements
        agent_prompt="Be expressive in your movements and use your hands while talking.",
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
