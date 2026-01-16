import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, inference
from livekit.plugins import silero

logger = logging.getLogger("realtime-scribe-v2")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT(
            model="elevenlabs/scribe_v2_realtime",
            language="en",
            extra_kwargs={
                "server_vad": {
                    "vad_silence_threshold_secs": 0.5,
                    "vad_threshold": 0.5,
                    "min_speech_duration_ms": 100,
                    "min_silence_duration_ms": 300,
                },
            },
        ),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="elevenlabs/eleven_turbo_v2_5"),
    )
    await session.start(
        agent=Agent(instructions="You are a somewhat helpful assistant."), room=ctx.room
    )

    await session.say("Hello, how can I help you?", allow_interruptions=False)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
