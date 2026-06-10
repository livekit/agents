import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import nvidia, openai

logger = logging.getLogger("basic-agent")

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.LLM(model="gpt-4.1-mini"),
        stt=nvidia.STT(),
        tts=nvidia.TTS(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,
    )

    await session.start(
        agent=Agent(instructions="You are a helpful voice AI assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
