"""Voice agent using LLMTurnDetector for end-of-turn classification.

Run with:
    OPENAI_API_KEY=... DEEPGRAM_API_KEY=... \
        python examples/voice_agents/llm_turn_detector.py dev
"""

from __future__ import annotations

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector import LLMTurnDetector


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # A cheap model is usually plenty for turn-completion classification.
    classifier_llm = openai.LLM(model="gpt-4o-mini")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        turn_detection=LLMTurnDetector(llm=classifier_llm),
    )

    await session.start(
        agent=Agent(instructions="You are a friendly assistant. Keep answers short."),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
