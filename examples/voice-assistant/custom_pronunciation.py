from __future__ import annotations
from typing import AsyncIterable
import asyncio

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, utils
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, cartesia, openai, silero

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    async def _will_synthesize_assistant_speech(
        assistant: VoiceAssistant, text: str | AsyncIterable[str]
    ):
        # Cartesia TTS is incorrectly pronouncing "LiveKit", so we'll replace it with a phonetic
        # spelling
        return utils.replace_words(
            text=text, replacements={"LiveKit": "<<L|ˈaɪ|ve>>Kit"}
        )

    # also for this example, we also intensify the keyword "LiveKit" to make it more likely to be
    # recognized with the STT
    deepgram_stt = deepgram.STT(keywords=[("LiveKit", 2.0)])

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram_stt,
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        will_synthesize_assistant_speech=_will_synthesize_assistant_speech,
    )
    assistant.start(ctx.room)

    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
