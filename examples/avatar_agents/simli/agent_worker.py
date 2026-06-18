import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai, simli

# from livekit.plugins import deepgram, elevenlabs, silero

logger = logging.getLogger("simli-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    simliAPIKey = os.getenv("SIMLI_API_KEY")
    simliFaceID = os.getenv("SIMLI_FACE_ID")

    simli_avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=simliAPIKey,
            face_id=simliFaceID,
        ),
    )
    await simli_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
