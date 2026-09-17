#!/usr/bin/env python3
# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example: Voice Agent with RNNoise Noise Cancellation

This example demonstrates how to integrate RNNoise noise cancellation
into a LiveKit voice agent for human-to-bot conversations. RNNoise runs
entirely on-device -- no API key, license, or model download required.

The audio pipeline:
    Room → RoomIO (with RNNoise) → VAD → STT → LLM → TTS → Room

Prerequisites:
    Install required packages:
       - livekit-agents
       - livekit-plugins-rnnoise
       - livekit-plugins-openai (or your preferred STT/LLM/TTS)

Usage:
    python rnnoise_agent_example.py dev
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
    room_io,
)
from livekit.plugins import openai, rnnoise

logger = logging.getLogger("rnnoise-agent-example")
load_dotenv()


class RNNoiseAgent(Agent):
    """Voice agent that uses RNNoise for noise cancellation."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Keep your responses concise and conversational. "
                "Do not use emojis or special characters in your responses."
            ),
        )

    async def on_enter(self) -> None:
        """Called when the agent enters the session."""
        logger.info("RNNoise agent entered session")
        self.session.generate_reply(allow_interruptions=False)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the agent session."""

    session = AgentSession(
        vad=inference.VAD(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )

    logger.info("Starting agent session with RoomIO and RNNoise noise cancellation")

    # Start the session with RoomIO configuration
    await session.start(
        agent=RNNoiseAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=rnnoise.RNNoise(),
            ),
        ),
    )

    logger.info("RNNoise noise cancellation active")


if __name__ == "__main__":
    cli.run_app(server)
