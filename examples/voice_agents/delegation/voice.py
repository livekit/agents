"""The voice half of a delegating agent: it talks, the fare desk thinks.

Start the expert first, in another terminal:

    python expert.py dev

then this, and call in:

    python voice.py console

Two processes on one machine, talking A2A over localhost. The voice agent has no tools of
its own beyond the one delegation gives it, and knows nothing about fares: it keeps the
caller company, sends the question to the expert, and phrases whatever comes back. Ask it
something that needs a lookup — "what would it cost to move my Monday flight to Tuesday,
and is there space?" — and you will hear it acknowledge, then report the expert's progress,
then answer.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    DelegationDirectiveEvent,
    JobContext,
    cli,
    inference,
)
from livekit.agents.delegation import A2ADelegate

logger = logging.getLogger("voice")

load_dotenv()

FARE_DESK_URL = "http://localhost:8321/fare-desk"

server = AgentServer()


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You answer the phone for Northwind Air. Keep the caller company and speak "
                "the way a person does: short sentences, no lists, no markdown. You do not "
                "know anything about fares, bookings or policy yourself — send every such "
                "question to the expert and say what comes back in your own words. Never "
                "mention the expert, and never promise an outcome before it answers."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="greet the caller as Northwind Air and ask how you can help"
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        # one delegate per conversation: the session closes it when the call ends
        delegate=A2ADelegate(FARE_DESK_URL),
    )

    @session.on("delegation_directive")
    def _on_directive(ev: DelegationDirectiveEvent) -> None:
        # advice, acted on after the answer has been said. What to do about it is yours:
        # here the caller is done, so the room is closed once the goodbye has played.
        logger.info("the expert advises %s (%s)", ev.kind, ev.reason)
        if ev.kind == "end_session":
            ctx.delete_room()

    await session.start(agent=Receptionist(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
