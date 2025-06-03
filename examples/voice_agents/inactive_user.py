import logging

from dotenv import load_dotenv

import asyncio
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    UserStateChangedEvent,
)
from livekit.plugins import deepgram, openai, cartesia, silero

logger = logging.getLogger("get-email-agent")

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        user_away_timeout=12.5,
    )

    inactivity_task: asyncio.Task | None = None

    async def user_presence_task():
        # try to ping the user 3 times, if we get no answer, close the session
        for i in range(3):
            await session.generate_reply(
                instructions=(
                    "The user has been inactive. Politely check if the user is still present, and "
                    "gently guide the conversation back toward your intended goal."
                )
            )
            await asyncio.sleep(10)

        await asyncio.shield(session.aclose())
        ctx.delete_room()

    @session.on("user_state_changed")
    def _user_state_changed(ev: UserStateChangedEvent):
        nonlocal inactivity_task
        if ev.new_state == "away":
            inactivity_task = asyncio.create_task(user_presence_task())
            return

        # ev.new_state: listening, speaking, ..
        if inactivity_task is not None:
            inactivity_task.cancel()

    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
