from __future__ import annotations

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero, spatius

logger = logging.getLogger("spatius-avatar-example")

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant.")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(),
    )

    spatius_avatar = spatius.AvatarSession()
    await spatius_avatar.start(session, room=ctx.room)

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
