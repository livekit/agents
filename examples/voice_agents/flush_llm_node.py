import asyncio
import logging
from collections.abc import AsyncIterable

import aiohttp
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FlushSentinel,
    JobContext,
    MetricsCollectedEvent,
    ModelSettings,
    cli,
    function_tool,
    llm,
    metrics,
)
from livekit.plugins import silero

logger = logging.getLogger("flush-llm-node")

load_dotenv()

## This example shows how to flush a fast response in `llm_node` to tts immediately.


class FastResponseAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
        )

    @function_tool
    async def get_weather(
        self,
        location: str,
        latitude: str,
        longitude: str,
    ):
        """Called when the user asks about the weather. This function will return the weather for
        the given location. When given a location, please estimate the latitude and longitude of the
        location and do not ask the user for them.

        Args:
            location: The location to get the weather for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """

        logger.info(f"getting weather for {location}")
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

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | FlushSentinel]:
        called_tools: list[llm.FunctionToolCall] = []
        has_text_message = False
        async for chunk in Agent.default.llm_node(
            agent=self,
            chat_ctx=chat_ctx,
            tools=tools,
            model_settings=model_settings,
        ):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta:
                if chunk.delta.content:
                    has_text_message = True
                if chunk.delta.tool_calls:
                    called_tools.extend(chunk.delta.tool_calls)

            yield chunk

        # example: fast response conditioned on the tool call name and the presence of a text message
        tool_names = [tool.name for tool in called_tools]
        if not has_text_message and "get_weather" in tool_names:
            logger.info("Fast response triggered")
            yield "One moment while I look that up. "
            # flush the response to tts immediately
            # NOTE: this will close the current tts_node and start a new one
            yield FlushSentinel()

            # simulate additional processing before closing the llm_node
            await asyncio.sleep(3)
            yield "Okay I found what you were looking for... "

        logger.info("LLM node completed")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm="openai/gpt-4.1-mini",
        stt="assemblyai/universal-streaming",
        tts="cartesia",
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=FastResponseAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
