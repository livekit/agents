import logging

from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.openai.beta import (
    AssistantCreateOptions,
    AssistantLLM,
    AssistantOptions,
    OnFileUploadedInfo,
)

load_dotenv()
logger = logging.getLogger("openai_assistant")


async def entrypoint(ctx: JobContext):
    """This example demonstrates a VoicePipelineAgent that uses OpenAI's Assistant API as the LLM"""
    initial_ctx = llm.ChatContext()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # When you add a ChatMessage that contain images, AssistantLLM will upload them
    # to OpenAI's Assistant API.
    # It's up to you to remove them if desired or otherwise manage them going forward.
    def on_file_uploaded(info: OnFileUploadedInfo):
        logger.info(f"{info.type} uploaded: {info.openai_file_object}")

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=AssistantLLM(
            assistant_opts=AssistantOptions(
                create_options=AssistantCreateOptions(
                    model="gpt-4o",
                    instructions="You are a voice assistant created by LiveKit. Your interface with users will be voice.",
                    name="KITT",
                )
            ),
            on_file_uploaded=on_file_uploaded,
        ),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
