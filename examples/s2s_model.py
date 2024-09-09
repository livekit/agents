from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
)
from typing import Annotated
from livekit.plugins import openai

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)



async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")
    
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable(description="Turn on/off the lights in a room")
    async def toggle_light(
        room: Annotated[str, llm.TypeInfo(description="The specific room")],
        status: bool,
    ):
        print(f"Turned the lights in the {room} {'on' if status else 'off'}")
        return f"Turned the lights in the {room} {'on' if status else 'off'}"

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = openai.VoiceAssistant(
        system_message="You are an artist that loves to sing",
        fnc_ctx=fnc_ctx,
    )
    assistant.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
