import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    CloseEvent,
    ErrorEvent,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,  # noqa: F401
    tts,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("basic-agent")

load_dotenv()


# This example demonstrates how to handle errors from STT, TTS, and LLM
# and how to continue the conversation after an error if the error is recoverable


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(
            use_realtime=False,  # set to True to use streaming STT
            # base_url="http://wrong-api-url",  # to simulate a failed connection
        ),
        tts=openai.TTS(voice="ash"),
    )

    @session.on("error")
    def _on_error(ev: ErrorEvent):
        if ev.error.recoverable:
            return

        # TTS and LLM errors can be marked as recoverable
        # since these components are recreated for each response
        if isinstance(ev.source, (tts.TTS, llm.LLM)):
            ev.error.recoverable = True
            return

        # STT stream persists for the entire agent lifetime
        # we can reset the agent if we want to continue the conversation
        # if isinstance(ev.source, stt.STT):
        #     session.update_agent(session.current_agent)
        #     ev.error.recoverable = True
        #     return

        # or the session will be closed after this event
        logger.warning("unrecoverable error, closing session")

    @session.on("close")
    def _on_close(ev: CloseEvent):
        logger.info("session closed")
        # (optional) stop the job when the session is closed
        ctx.shutdown()

    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
