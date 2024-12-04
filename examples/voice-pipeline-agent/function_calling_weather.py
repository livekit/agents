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
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

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
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""

        # Example of a filler message while waiting for the function call to complete.
        # NOTE: This message illustrates how the agent can engage users by using the `say()` method
        # while awaiting the completion of the function call. To create a more dynamic and engaging
        # interaction, consider varying the responses based on context or user input.
        call_ctx = AgentCallContext.get_current()
        # message = f"Let me check the weather in {location} for you."
        message = f"Here is the weather in {location}: "
        filler_messages = [
            "Let me check the weather in {location} for you.",
            "Let me see what the weather is like in {location} right now.",
            # LLM will complete this sentence if it is added to the end of the chat context
            "The current weather in {location} is ",
        ]
        message = random.choice(filler_messages).format(location=location)

        # NOTE: set add_to_chat_ctx=True will add the message to the end
        #   of the chat context of the function call for answer synthesis
        speech_handle = await call_ctx.agent.say(message, add_to_chat_ctx=True)  # noqa: F841

        # To wait for the speech to finish
        # await speech_handle.join()

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


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are a weather assistant created by LiveKit. Your interface with users will be voice. "
            "You will provide weather information for a given location."
        ),
        role="system",
    )
    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
    )
    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say(
        "Hello from the weather station. Would you like to know the weather? If so, tell me your location."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
