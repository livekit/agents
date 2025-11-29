import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import elevenlabs, openai, silero

logger = logging.getLogger("realtime-scribe-v2")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    stt = elevenlabs.STT(
        use_realtime=True,
        server_vad={
            "vad_silence_threshold_secs": 0.5,
            "vad_threshold": 0.5,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 300,
        },
    )

    session = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=elevenlabs.TTS(model="eleven_turbo_v2_5"),
    )
    await session.start(
        agent=Agent(instructions="You are a somewhat helpful assistant."), room=ctx.room
    )

    await session.say("Hello, how can I help you?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
