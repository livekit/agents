import asyncio
import logging
import random
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, elevenlabs, openai, silero

load_dotenv()

logger = logging.getLogger("weather-demo")
logger.setLevel(logging.INFO)


class AssistantFnc(llm.FunctionContext):
    """
    The class defines a set of LLM functions that the assistant can execute.
    """

    @llm.ai_callable()
    async def get_weather(
        self,
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
        latitude: Annotated[
            str,
            llm.TypeInfo(description="The latitude of location to get the weather for"),
        ],
        longitude: Annotated[
            str,
            llm.TypeInfo(
                description="The longitude of location to get the weather for"
            ),
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location.
        When given a location, please estimate the latitude and longitude of the location and do not ask the user for them."""

        # When a function call is running, there are a couple of options to inform the user
        # that it might take awhile:
        # Option 1: you can use .say filler message immediately after the call is triggered
        # Option 2: you can prompt the agent to return a text response when it's making a function call
        agent = AgentCallContext.get_current().agent

        if (
            not agent.chat_ctx.messages
            or agent.chat_ctx.messages[-1].role != "assistant"
        ):
            # skip if assistant already said something
            filler_messages = [
                "Let me check the weather in {location} for you.",
                "Let me see what the weather is like in {location} right now.",
                # LLM will complete this sentence if it is added to the end of the chat context
                "The current weather in {location} is ",
            ]
            message = random.choice(filler_messages).format(location=location)
            logger.info(f"saying filler message: {message}")

            # NOTE: set add_to_chat_ctx=True will add the message to the end
            #   of the chat context of the function call for answer synthesis
            # speech_handle = await agent.say(message, add_to_chat_ctx=True)  # noqa: F841

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
                    raise Exception(
                        f"Failed to get weather data, status code: {response.status}"
                    )

        await asyncio.sleep(3)
        logger.info(f"weather data: {weather_data}")

        # (optional) To wait for the speech to finish before giving results of the function call
        # await speech_handle.join()
        return weather_data


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are a weather assistant created by LiveKit. Your interface with users will be voice. "
            "You will provide weather information for a given location. "
            # when using option 1, you can suppress from the agent with prompt
            # "do not return any text while calling the function."
            # uncomment this to use option 2
            "when performing function calls, let user know that you are checking the weather."
        ),
        role="system",
    )
    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice=elevenlabs.tts.Voice(
                id="XrExE9yKIg1WjnnlVkGX",
                name="Matilda",
                category="premade",
                settings=elevenlabs.tts.VoiceSettings(
                    stability=0.8,
                    similarity_boost=0.8,
                    style=0.0,
                    use_speaker_boost=True,
                    # speed=0.95,
                ),
            ),
            language="en",
            enable_ssml_parsing=False,
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say(
        "Let me just check my system. Thank you for agreeing to pay $100. However, your payment must be between $418.63 and $1012.11."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
