from __future__ import annotations

import logging
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    multimodal,
)
from livekit.plugins import google

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def get_weather(
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise Exception(
                        f"Failed to get weather data, status code: {response.status}"
                    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    chat_ctx = llm.ChatContext()

    agent = multimodal.MultimodalAgent(
        model=google.beta.realtime.RealtimeModel(
            voice="Charon",
            temperature=0.8,
            instructions="You are a helpful assistant",
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
