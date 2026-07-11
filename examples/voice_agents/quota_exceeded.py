import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIQuotaExceededError,
    JobContext,
    cli,
    inference,
)
from livekit.agents.voice.events import CloseEvent, ErrorEvent
from livekit.plugins import silero

logger = logging.getLogger("quota-exceeded")

load_dotenv()

# This example shows how to keep a voice agent from going *silently* unresponsive
# when the LLM endpoint returns `429 inference_quota_exceeded` (e.g. the project ran
# out of LiveKit Inference credits).
#
# Without this handling, such an error used to make the agent join the room, publish
# its track, and then never speak. The SDK now surfaces a terminal quota error on the
# FIRST occurrence (instead of after several dead turns) and, by default, speaks the
# gateway's own `hint` before the session closes. This example builds on that:
#
#   1. `error_message=...` replaces the default spoken line with your own message
#      (omit it to keep speaking the quota `hint`; pass None to disable spoken errors).
#
#   2. The `@session.on("error")` handler shows how to read the typed
#      `APIQuotaExceededError` (status_code, quota_type, hint, ...) so you can forward
#      a structured "out of credits" state to your frontend.

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(),
        # spoken just before the session closes on an unrecoverable error so the agent
        # is never silent. Omit this argument entirely to keep the default behavior
        # (speak the quota `hint`); pass None to disable spoken errors.
        error_message="Sorry, the assistant is temporarily unavailable. Please try again later.",
    )

    @session.on("error")
    def on_error(ev: ErrorEvent) -> None:
        # ErrorEvent.error is the LLMError/STTError/TTSError wrapper; the underlying
        # API exception is at ev.error.error
        err = ev.error.error
        # this handler also sees transient errors (e.g. rate limits, including retry
        # attempts); only a *terminal* quota error means the project is out of credits
        # and will fail identically every turn until the quota resets
        if isinstance(err, APIQuotaExceededError) and err.terminal:
            logger.warning(
                "inference quota exceeded",
                extra={
                    "quota_type": err.quota_type,  # "llm" | "stt" | "tts" | ...
                    "category": err.category,  # e.g. "MaxGatewayCredits"
                    "hint": err.hint,
                    "remaining_limit": err.remaining_limit,
                },
            )
            # forward a structured signal so the frontend can render an
            # "out of credits" state instead of dead air. `session.on` handlers are
            # sync, so spawn a task for async work (add `import asyncio` above), e.g.:
            #
            # asyncio.create_task(
            #     ctx.room.local_participant.set_attributes(
            #         {"agent_error": "quota_exceeded", "quota_type": err.quota_type or ""}
            #     )
            # )

    @session.on("close")
    def on_close(ev: CloseEvent) -> None:
        logger.info("session closed", extra={"reason": ev.reason})

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
