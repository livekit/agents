import logging

from dotenv import load_dotenv

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentInterruptionResumedEvent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("resume-agent")

load_dotenv()

# This example shows how to resume an agent from a false interruption.
# If `interruption_resume_delay` is set, AgentSession will emit an
# `AgentInterruptionResumeEvent` event when the agent is interrupted by the user audio
# but there is no new user transcript after the interruption_resume_delay.


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        interruption_resume_delay=4.0,
    )

    @session.on("agent_interruption_resumed")
    def _agent_interruption_resumed(ev: AgentInterruptionResumedEvent):
        if ev.old_speech_source != "say":
            logger.info(
                "Resuming agent from interruption",
                extra={"instructions": ev.old_instructions, "forwarded_text": ev.forwarded_text},
            )
            instructions = ev.old_instructions or NOT_GIVEN
            session.generate_reply(instructions=instructions)

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
