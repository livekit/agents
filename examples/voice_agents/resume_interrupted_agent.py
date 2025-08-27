import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("resume-agent")

load_dotenv()

# This example shows how to resume an agent from a false interruption.
# If `resume_false_interruption` is True, the agent will first pause the audio output
# while not interrupting the speech before the `false_interruption_timeout` expires.
# If there is not new user input after the pause, the agent will resume the output for the same speech.
# If there is new user input, the agent will interrupt the speech immediately.


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        false_interruption_timeout=1.0,
        resume_false_interruption=True,
    )

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
