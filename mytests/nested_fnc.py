import logging
import random
import re
import urllib
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
    async def get_city_code(
        self,
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the city code for")
        ],
    ) -> str:
        """Called when the user asks about the weather of a city. This function will return the city code for the given location for next steps."""

        logger.info(f"getting city code for {location}")
        return f"code-{location}"

    @llm.ai_callable()
    async def get_weather(
        self,
        city_code: Annotated[
            str, llm.TypeInfo(description="The city code to get the weather for")
        ],
    ):
        """Called after the city code is returned for getting the weather."""
        # Clean the location string of special characters
        if city_code.startswith("code-"):
            location = city_code[5:]
        else:
            location = city_code

        location = re.sub(r"[^a-zA-Z0-9]+", " ", location).strip()

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
            speech_handle = await agent.say(message, add_to_chat_ctx=True)  # noqa: F841

        logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=%C+%t"
        weather_data = ""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    # response from the function call is returned to the LLM
                    weather_data = (
                        f"The weather in {location} is {await response.text()}."
                    )
                    logger.info(f"weather data: {weather_data}")
                else:
                    raise Exception(
                        f"Failed to get weather data, status code: {response.status}"
                    )

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
            "You will first get the city code for the given location, then get the weather for the city code."
            # when using option 1, you can suppress from the agent with prompt
            "do not return any text while calling the function."
            # uncomment this to use option 2
            # "when performing function calls, let user know that you are checking the weather."
        ),
        role="system",
    )
    participant = await ctx.wait_for_participant()

    async def before_llm_cb(agent: VoicePipelineAgent, chat_ctx):
        logger.info(f"before llm cb, {agent._playing_speech}")
        speech_handle = await agent.say(
            "I am checking the weather for you.", add_to_chat_ctx=True
        )
        llm_stream = agent.llm.chat(chat_ctx=chat_ctx, fnc_ctx=agent.fnc_ctx)
        # await speech_handle.join()
        return llm_stream

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        max_nested_fnc_calls=2,
        before_llm_cb=before_llm_cb,
    )

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)

    @agent.on("agent_speech_committed")
    def on_speech_committed(*args, **kwargs):
        import pprint

        pprint.pprint(agent.chat_ctx.messages)

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
