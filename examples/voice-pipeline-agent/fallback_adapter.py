import logging
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, playai, silero

load_dotenv()
logger = logging.getLogger("fallback-adapter-example")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    def get_time():
        """called to retrieve the current local time"""
        return datetime.now().strftime("%H:%M:%S")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    vad: silero.VAD = ctx.proc.userdata["vad"]

    # fallback to OpenAI if Deepgram goes down
    fallback_stt = stt.FallbackAdapter(
        [
            deepgram.STT(),
            stt.StreamAdapter(stt=openai.STT(), vad=vad),
        ]
    )

    # fallback to Azure if OpenAI goes down
    fallback_llm = llm.FallbackAdapter(
        [
            openai.LLM(),
            openai.LLM.with_azure(),
        ]
    )

    # fallback to 11labs if Cartesia goes down
    # you can keep the same voice by using their voice cloning feature
    fallback_tts = tts.FallbackAdapter(
        [
            cartesia.TTS(),
            playai.TTS(),
        ]
    )

    agent = VoicePipelineAgent(
        vad=vad,
        stt=fallback_stt,
        llm=fallback_llm,
        tts=fallback_tts,
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    agent.start(ctx.room, participant)

    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
