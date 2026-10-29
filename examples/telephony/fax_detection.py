import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FaxDetector,
    JobContext,
    cli,
    inference,
)

logger = logging.getLogger("fax-detection")

load_dotenv()

# Inbound phone agents sometimes get calls from fax machines: a beep every ~3.5 s and no
# speech. These calls can't be helped and skew call metrics. This example tags and hangs up
# on them. Dispatch inbound SIP calls to this agent; see
# https://docs.livekit.io/telephony/accepting-calls/


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a friendly receptionist answering phone calls for a clinic.",
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    # wait for the caller so the detector listens to the right participant
    caller = await ctx.wait_for_participant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)

    await session.start(agent=MyAgent(), room=ctx.room)
    session.generate_reply(instructions="Greet the caller and ask how you can help.")

    # the conversation runs in the background while we listen for a fax tone,
    # so a human caller is never delayed
    async with FaxDetector(ctx.room, participant_identity=caller.identity) as detector:
        result = await detector.wait()

    if result is not None:
        logger.info("fax machine detected, ending call", extra={"after": result.detected_after})
        # tag the call so analytics can exclude it, then hang up
        await ctx.room.local_participant.set_attributes({"call.fax_detected": "true"})
        ctx.delete_room()


if __name__ == "__main__":
    cli.run_app(server)
