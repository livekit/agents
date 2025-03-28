import logging
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import google

logger = logging.getLogger("weather-example")
logger.setLevel(logging.INFO)

load_dotenv()


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a weather agent.",
            llm=google.beta.realtime.RealtimeModel(),
        )

    @function_tool
    async def get_weather(
        self,
        location: Annotated[str, Field(description="The location to get the weather for")],
        latitude: Annotated[
            str, Field(description="The latitude of location to get the weather for")
        ],
        longitude: Annotated[
            str, Field(description="The longitude of location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for
        the given location. When given a location, please estimate the latitude and longitude of the
        location and do not ask the user for them.
        """

        logger.info(f"getting weather for {latitude}, {longitude}")
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        weather_data = {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # response from the function call is returned to the LLM
                    weather_data = {
                        "temperature": data["current"]["temperature_2m"],
                        "temperature_unit": "Celsius",
                    }
                else:
                    raise Exception(f"Failed to get weather data, status code: {response.status}")

        return weather_data


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
