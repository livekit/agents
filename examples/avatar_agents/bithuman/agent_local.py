import logging
import os

from bithuman import AsyncBithuman
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.plugins import bithuman, deepgram, openai, silero

logger = logging.getLogger("bithuman-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

bithuman_model_path = os.getenv("BITHUMAN_MODEL_PATH")
bithuman_api_secret = os.getenv("BITHUMAN_API_SECRET")


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="ash"),
        resume_false_interruption=True,
        min_interruption_duration=0.2,
    )

    logger.info("starting bithuman runtime")
    bithuman_avatar = bithuman.AvatarSession(
        model_path=bithuman_model_path,
        api_secret=bithuman_api_secret,
        runtime=ctx.proc.userdata.get("bithuman_runtime"),
    )
    await bithuman_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Your are Einstein, talk to me!"),
        room=ctx.room,
    )


def prewarm(proc: JobProcess):
    if not bithuman_model_path:
        return

    # if we know the model path before job received, prewarm the runtime
    logger.info("loading bithuman runtime")
    runtime = AsyncBithuman(
        model_path=bithuman_model_path,
        api_secret=bithuman_api_secret,
        load_model=True,
        input_buffer_size=1,  # queue will be cleared when pause is called
    )
    logger.info("bithuman runtime loaded")
    proc.userdata["bithuman_runtime"] = runtime


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
            job_memory_warn_mb=1500,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
            num_idle_processes=1,
        )
    )
