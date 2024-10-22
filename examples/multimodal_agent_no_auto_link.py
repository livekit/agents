
from __future__ import annotations

import logging
from typing import Annotated

import aiohttp
import os
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    multimodal,
)
from livekit.plugins import openai

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env.local")
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are a helpful assistant, managing multiple participants in an audio room",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
        ),
        auto_link_on_connect=False
    )

    def on_participant_connected(participant: rtc.Participant):
        identity = participant.identity
        agent._link_participant(identity)
        logger.info(f"Participant connected: {identity}")

    ctx.room.on("participant_connected", on_participant_connected)

    agent.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
