import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("resume-agent")

load_dotenv()

# This example shows how to resume an agent from a false interruption.
# If `resume_false_interruption` is True, the agent will resume from the interruption using the default callback
# that uses `generate_reply` to resume the agent.
# Or it can be a custom callback that will be called with the session and the event.


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        agent_false_interruption_timeout=3.0,
        resume_false_interruption=True,  # it can be a custom callback as well
    )

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
