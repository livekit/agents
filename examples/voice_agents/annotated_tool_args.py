import logging
from enum import Enum
from typing import Annotated, Literal  # noqa: F401

from dotenv import load_dotenv
from pydantic import Field  # noqa: F401

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("annotated-tool-args")
logger.setLevel(logging.INFO)

load_dotenv()


## This example demonstrates how to use function tools with type hints and descriptions
## The Args in docstring will be parsed as arg descriptions for the LLM
## You can also use enums and pydantic.Field to add descriptions
## For dynamic tool creation, check out dynamic_tool_creation.py


class RoomName(str, Enum):
    BEDROOM = "bedroom"
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"
    GARAGE = "garage"


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=("You are a helpful assistatn."),
        )

    @function_tool
    async def get_weather(self, location: str) -> str:
        """
        Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """

        # LLM will see location as a string argument with the description defined in docstring
        # {
        #     "description": "The location to get the weather for"
        #     "title": "Location"
        #     "type": "string",
        # }

        # Another way to add descriptions to the arguments
        # location: Annotated[str, Field(description="The location to get the weather for")]

        logger.info(f"Getting weather for {location}")
        return f"The weather in {location} is sunny today."

    @function_tool
    async def toggle_light(self, room: RoomName, switch_to: Literal["on", "off"]) -> str:
        """
        Called when the user asks to turn on or off the light.

        Args:
            room: The room to turn the light in
            switch_to: The state to turn the light to
        """

        logger.info(f"Turning light to {switch_to} in {room}")
        return f"The light in the {room.value} is now {switch_to}."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )

    await agent.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
