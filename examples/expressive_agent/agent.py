import logging
from pathlib import Path

from dotenv import load_dotenv
from protocol import SessionRequest

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)

logger = logging.getLogger("expressive-agent")

load_dotenv()

AGENT_NAME = "expressive_agent"
INSTRUCTIONS = (Path(__file__).parent / "prompt.md").read_text()

GREETING = (
    "Open the call the way you'd answer the phone to someone you know well. "
    "Short and warm, and leave them room to say what's going on."
)


class Friend(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions=GREETING)


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def expressive_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    config = SessionRequest.parse(ctx.job.metadata).resolve()
    logger.info(
        "starting session",
        extra={"expressive": config.expressive, "voice": config.voice.label},
    )

    session = AgentSession(
        stt=inference.STT("assemblyai/universal-3-5-pro", language="en"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS(config.voice.model, voice=config.voice.voice),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1"),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
        expressive=config.expressive,
    )

    await session.start(agent=Friend(), room=ctx.room)
    await ctx.connect()

    await ctx.room.local_participant.set_attributes(config.attributes())


if __name__ == "__main__":
    cli.run_app(server)
