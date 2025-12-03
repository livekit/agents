"""
Example LiveKit Agent using WebSocket LLM Provider

This example demonstrates how to use the custom WebSocket LLM provider
with a LiveKit voice agent. It connects to the WebSocket LLM server
instead of using standard providers like OpenAI.

Prerequisites:
    1. Start the WebSocket LLM server:
       uv run ws_llm_server.py
       
    2. Run this agent:
       uv run agent_example.py dev

The agent will:
- Listen for voice input via STT (Deepgram)
- Send text to WebSocket LLM server for responses
- Speak responses via TTS (Cartesia)
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from ws_llm_provider import WebSocketLLM

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    metrics,
)
from livekit.agents.worker import AgentServer
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("websocket-llm-agent")
logger.setLevel(logging.INFO)


class HealthcareAgent(Agent):
    """A healthcare assistant agent using WebSocket LLM backend."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a healthcare assistant. Help users with health-related questions. "
                "Keep responses concise and suitable for voice interaction. "
                "Do not use markdown, emojis, or special formatting in responses."
            ),
        )

    async def on_enter(self):
        """Called when the agent starts - generate initial greeting."""
        self.session.generate_reply(
            instructions="Greet the user and introduce yourself as a healthcare assistant. "
            "Ask how you can help them today. Keep it brief."
        )


# Create the agent server
server = AgentServer()


def prewarm(proc: JobProcess):
    """Pre-warm resources that can be shared across sessions."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entry point for each agent session."""
    logger.info(f"Starting agent session for room: {ctx.room.name}")

    # Create the WebSocket LLM provider
    # Make sure ws_llm_server.py is running on localhost:8765
    ws_llm = WebSocketLLM(
        ws_url="ws://localhost:8765",
        model_name="healthcare-assistant",
    )

    # Create the agent session with all components
    session = AgentSession(
        # Speech-to-Text: Convert user speech to text
        stt=deepgram.STT(model="nova-3"),
        # LLM: Our custom WebSocket LLM provider
        llm=ws_llm,
        # Text-to-Speech: Convert LLM responses to speech
        tts=cartesia.TTS(voice="79a125e8-cd45-4c13-8a67-188112f4dd22"),  # Default voice
        # Voice Activity Detection
        vad=ctx.proc.userdata["vad"],
        # Turn detection for natural conversation flow
        turn_detection=MultilingualModel(),
    )

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Session usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Connect to the room and start the session
    await ctx.connect()
    await session.start(
        agent=HealthcareAgent(),
        room=ctx.room,
    )

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(server)

