from __future__ import annotations

from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, tokenize
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "  # noqa: E501
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."  # noqa: E501
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    def _before_tts_cb(agent: VoicePipelineAgent, text: str | AsyncIterable[str]):
        # The TTS is incorrectly pronouncing "LiveKit", so we'll replace it with a phonetic
        # spelling
        return tokenize.utils.replace_words(
            text=text, replacements={"livekit": r"<<l|aɪ|v|k|ɪ|t|>>"}
        )

    # also for this example, we also intensify the keyword "LiveKit" to make it more likely to be
    # recognized with the STT
    deepgram_stt = deepgram.STT(keywords=[("LiveKit", 3.5)])

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram_stt,
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        before_tts_cb=_before_tts_cb,
    )
    agent.start(ctx.room)

    await agent.say("Hey, LiveKit is awesome!", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
