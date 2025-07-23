import logging

from dotenv import load_dotenv

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("resume-agent")

load_dotenv()

# This example shows how to resume an agent from a false interruption.
# If `speech_resume_delay` is set, AgentSession will emit an `AgentSpeechResumeEvent` event
# when the agent is interrupted by the user audio but there is no new user transcript
# after the speech_resume_delay.


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        agent_false_interruption_timeout=3.0,
    )

    @session.on("agent_false_interruption")
    def _agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info(
            "Resuming agent from interruption", extra={"instructions": ev.extra_instructions}
        )
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
