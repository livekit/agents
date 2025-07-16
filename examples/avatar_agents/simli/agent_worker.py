import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import openai, simli

# from livekit.plugins import deepgram, elevenlabs, silero

logger = logging.getLogger("simli-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )
    # session = AgentSession(
    #     vad=silero.VAD.load(),
    #     llm=openai.LLM(model="gpt-4.1-mini"),
    #     stt=deepgram.STT(model="nova-3"),
    #     tts=elevenlabs.TTS(model="eleven_flash_v2_5"),
    #     user_away_timeout=12.5,
    # )

    simliAPIKey = os.getenv("SIMLI_API_KEY")
    simliFaceID = os.getenv("SIMLI_FACE_ID")

    simli_avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            apiKey=simliAPIKey,
            faceId=simliFaceID,
        ),
        api_url="http://127.0.0.1:6069",
    )
    await simli_avatar.start(session, room=ctx.room)

    # start the agent, it will join the room and wait for the avatar to join
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
