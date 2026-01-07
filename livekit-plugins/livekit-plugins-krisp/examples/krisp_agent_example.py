#!/usr/bin/env python3
"""
Example: Voice Agent with Krisp Noise Cancellation

This example demonstrates how to integrate Krisp noise cancellation
into a LiveKit voice agent for human-to-bot conversations.

The audio pipeline:
    Room → RoomIO (with KrispVivaFilterFrameProcessor) → VAD → STT → LLM → TTS → Room

Prerequisites:
    1. Set KRISP_VIVA_FILTER_MODEL_PATH environment variable to your .kef model file
    2. Install required packages:
       - livekit-agents (with PR #4145 support for FrameProcessor)
       - livekit-plugins-krisp
       - livekit-plugins-silero (for VAD)
       - livekit-plugins-openai (or your preferred STT/LLM/TTS)

Usage:
    python krisp_agent_example.py dev
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    room_io,
)
from livekit.plugins import krisp, openai, silero

logger = logging.getLogger("krisp-agent-example")
load_dotenv()


class KrispAgent(Agent):
    """Voice agent that uses Krisp for noise cancellation."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Keep your responses concise and conversational. "
                "Do not use emojis or special characters in your responses."
            ),
        )

    async def on_enter(self):
        """Called when the agent enters the session."""
        logger.info("Krisp agent entered session")
        # Generate initial greeting (uninterruptible for AEC calibration)
        self.session.generate_reply(allow_interruptions=False)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent session."""

    # Configure the agent session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )

    logger.info("Starting agent session with RoomIO and Krisp noise cancellation")

    # Create Krisp FrameProcessor for noise cancellation
    processor = krisp.KrispVivaFilterFrameProcessor(
        noise_suppression_level=100,  # 0-100, where 100 is maximum suppression
        frame_duration_ms=10,
        sample_rate=16000,  # Pre-load model at this sample rate
    )

    # Start the session with RoomIO configuration
    # IMPORTANT: frame_size_ms must match Krisp's frame_duration_ms
    await session.start(
        agent=KrispAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=16000,  # Krisp supports: 8k, 16k, 24k, 32k, 44.1k, 48k
                num_channels=1,
                frame_size_ms=10,  # Must match Krisp frame_duration_ms (10, 15, 20, 30, or 32)
                noise_cancellation=processor,  # Pass FrameProcessor directly
            ),
        ),
    )

    logger.info("✅ Krisp noise cancellation active")


if __name__ == "__main__":
    cli.run_app(server)
