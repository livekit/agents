from __future__ import annotations

import logging
import asyncio
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
    omni_assistant,
)
from livekit.plugins.openai import realtime

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=("You are a helpful assistant who will perform tasks as requested"),
    )

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
                print(response)
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise Exception(
                        "Failed to get weather data, status code: {response.status}"
                    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    model = realtime.RealtimeModel()
    assistant = omni_assistant.OmniAssistant(
        model=model,
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assistant.start(ctx.room)

    fnc_ctx = llm.FunctionContext()


    # e.g, parralel sentiment analysis

    @fnc_ctx.ai_callable()
    async def sentiment(
        sentiment: Annotated[
            str, llm.TypeInfo(description="the sentiment of the user based on his speech (be explicit)")
        ],
    ):
        print(sentiment)

    session = model.sessions[0]
    conv = session.create_conversation(label="sentiment-analysis", fnc_ctx=fnc_ctx,
                                       tool_choice="required")

    #while True:
    #    await asyncio.sleep(5)
    #    conv.generate()



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
