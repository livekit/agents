import logging
from enum import Enum
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("parallel-functions")
logger.setLevel(logging.INFO)

load_dotenv()


## This example demonstrates how to use function calls with type hints and descriptions
## The Args in docstring will be parsed as arg descriptions for the LLM
## You can also use enums and pydantic.Field to add descriptions


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
        self._my_location = "San Francisco"

    @function_tool
    async def get_weather(self, location: str) -> str:
        """
        Called when the user asks about the weather.

        Args:
            location: The location to get the weather for
        """

        # the LLM will see location as a string argument with a description
        # {
        #     "description": "The location to get the weather for"
        #     "title": "Location"
        #     "type": "string",
        # }
        logger.info(f"Getting weather for {location}")
        return f"The weather in {location} is sunny today."

    @function_tool
    async def toggle_light(
        self,
        room: Annotated[RoomName, Field(description="The room to turn the light in")],
        switch_to: Annotated[
            Literal["on", "off"], Field(description="The state to turn the light to")
        ],
    ) -> str:
        """
        Called when the user asks to turn on or off the light.
        """

        # The room argument will be parsed as a RoomName enum
        logger.info(f"Turning light to {switch_to} in {room}")
        return f"The light is now {switch_to}."


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
