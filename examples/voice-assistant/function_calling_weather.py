import logging
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import anthropic, deepgram, openai, silero

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
        print("NEIL get_weather")
        return "The weather is sunny."


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def handle_participant(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("NEIL handle_participant")
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are a weather assistant created by LiveKit. Your interface with users will be voice. "
            "You will provide weather information for a given location."
        ),
        role="system",
    )
    logger.info("NEIL Starting voice assistant 0")
    allm = anthropic.LLM()
    logger.info("NEIL Starting voice assistant")
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=allm,
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
    )
    # Start the assistant. This will automatically publish a microphone track and listen to the first participant
    # it finds in the current room. If you need to specify a particular participant, use the participant parameter.
    assistant.start(ctx.room, participant=participant)
    await assistant.say("Hello from the weather station.")


async def entrypoint(ctx: JobContext):
    print("NEILLL")
    ctx.add_participant_entrypoint(handle_participant)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
