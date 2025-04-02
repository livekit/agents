import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai

logger = logging.getLogger("text-only")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates a text-only agent.
## Send text input using TextStream to topic `lk.chat` (https://docs.livekit.io/home/client/data/text-streams)
## The agent output is sent through TextStream to the `lk.transcription` topic


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            llm=openai.LLM(model="gpt-4o-mini"),
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(text_enabled=True, audio_enabled=False),
        room_output_options=RoomOutputOptions(transcription_enabled=True, audio_enabled=False),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
