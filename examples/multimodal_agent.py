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
from livekit.plugins import openai

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

    # to use Microsoft Azure, uncomment the following lines
    # agent = multimodal.MultimodalAgent(
    #     model=openai.realtime.RealtimeModel.with_azure(
    #         azure_deployment="<model-deployment>",
    #         azure_endpoint="wss://<endpoint>.openai.azure.com/", # or AZURE_OPENAI_ENDPOINT
    #         api_key="<api-key>", # or AZURE_OPENAI_API_KEY
    #         api_version="2024-10-01-preview", # or OPENAI_API_VERSION
    #         voice="alloy",
    #         temperature=0.8,
    #         instructions="You are a helpful assistant",
    #         turn_detection=openai.realtime.ServerVadOptions(
    #             threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
    #         ),
    #     ),
    #     fnc_ctx=fnc_ctx,
    # )

    # create a chat context with chat history
    chat_ctx = llm.ChatContext()

    # # Add some test context to verify if the sync_chat_ctx works
    # chat_ctx.append(
    #     text="Hi! I'm Alice and I love hiking in the mountains.", role="user"
    # )
    # chat_ctx.append(
    #     text="Hello Alice! That's wonderful. I also enjoy outdoor activities and would be happy to discuss hiking adventures with you.",
    #     role="assistant",
    # )
    # chat_ctx.append(text="I'm planning a trip to Paris next month.", role="user")
    # chat_ctx.append(
    #     text="How exciting! Paris is a beautiful city. I'd be happy to suggest some must-visit places and help you plan your trip.",
    #     role="assistant",
    # )

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are a helpful assistant",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)

    @agent.on("agent_speech_committed")
    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        # example of truncating the chat context
        max_ctx_len = 10
        if len(agent.chat_ctx.messages) > max_ctx_len:
            messages = agent.chat_ctx.messages[-max_ctx_len:]
            agent.sync_chat_ctx(llm.ChatContext(messages=messages))

    @ctx.room.on("participant_attributes_changed")
    def _on_participant_attributes_changed(changed_attrs: dict[str, str]):
        if "lk.agent.state" in changed_attrs:
            agent_state = changed_attrs["lk.agent.state"]
            logger.info("agent state changed", extra={"agent_state": agent_state})


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
