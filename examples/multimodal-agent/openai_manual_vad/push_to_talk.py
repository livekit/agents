from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    JobRequest,
    WorkerType,
    cli,
    llm,
    multimodal,
)
from livekit.plugins import openai

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are a helpful assistant",
            turn_detection=None,
        ),
    )
    agent.start(ctx.room, participant)

    @ctx.room.local_participant.register_rpc_method("ptt")
    async def handle_ptt(data):
        logger.info(f"Received PTT action: {data.payload}")
        if data.payload == "push":
            agent.interrupt()
        elif data.payload == "release":
            agent.commit_audio_buffer()
        return "ok"

    @agent.on("agent_speech_committed")
    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        # example of truncating the chat context
        max_ctx_len = 10
        chat_ctx = agent.chat_ctx_copy()
        if len(chat_ctx.messages) > max_ctx_len:
            chat_ctx.messages = chat_ctx.messages[-max_ctx_len:]
            # NOTE: The `set_chat_ctx` function will attempt to synchronize changes made
            # to the local chat context with the server instead of completely replacing it,
            # provided that the message IDs are consistent.
            asyncio.create_task(agent.set_chat_ctx(chat_ctx))


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # This attribute communicates to frontend that we support PTT
        attributes={"supports-ptt": "1"},
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=handle_request,
        )
    )
