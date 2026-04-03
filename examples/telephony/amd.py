import logging

from dotenv import load_dotenv

from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv("../agents/.env")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are reaching out to a customer with a phone call. "
                "You are calling to see if they are home. "
                "You might encounter an answering machine with a DTMF menu or IVR system. "
                "If you do, you will try to leave a message to ask them to call back."
            ),
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
    )

    async with AMD(session, llm="openai/gpt-5-mini") as detector:
        result = await detector.execute()

        if result.category == "human":
            logger.info("human answered the call, proceeding with normal conversation")
        elif result.category == "machine-ivr":
            logger.info("ivr menu detected, starting navigation")
        elif result.category == "machine-vm":
            logger.info("voicemail detected, leaving a message")
            speech_handle = session.generate_reply(
                instructions=(
                    "You've reached voicemail. Leave a brief message asking "
                    "the customer to call back."
                ),
            )
            await speech_handle.wait_for_playout()
            session.shutdown()
        elif result.category == "machine-unavailable":
            logger.info("mailbox unavailable, ending call")
            session.shutdown()


if __name__ == "__main__":
    cli.run_app(server)
