"""
Basic Telnyx SIP Trunk Provider Example

This example demonstrates how to configure LiveKit Agents to work with Telnyx
as a SIP trunk provider for telephony integrations.

Requirements:
- LiveKit Cloud account with SIP trunk configured using Telnyx
- Telnyx account with phone number and SIP connection
- Environment variables set (see README.md)

Usage:
    python basic_telnyx_agent.py console
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice.events import RunContext
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("telnyx-agent")

load_dotenv()

# Telnyx SIP Trunk Configuration
# These environment variables should be set in your .env file
# See README.md for configuration instructions
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER", "")
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER", "")


class TelnyxAgent(Agent):
    """Example agent configured for Telnyx SIP trunk integration."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly customer service agent speaking with a caller "
                "over the phone via Telnyx SIP trunking. "
                "Keep your responses concise, warm, and professional. "
                "Speak clearly and at a measured pace. "
                "Introduce yourself and ask how you can help."
            ),
        )

    async def on_enter(self) -> None:
        """Called when the agent joins the call."""
        self.session.generate_reply(
            instructions=(
                "Greet the caller warmly, introduce yourself as a customer service "
                "representative, and ask how you can help them today."
            ),
            allow_interruptions=False,
        )

    @function_tool
    async def get_store_hours(self, context: RunContext) -> str:
        """Get the current store hours for the business."""
        return (
            "Our store is open Monday through Friday, 9 AM to 6 PM, "
            "and Saturday, 10 AM to 4 PM. We're closed on Sundays."
        )

    @function_tool
    async def get_location(self, context: RunContext) -> str:
        """Get the physical location of the business."""
        return (
            "We are located at 123 Main Street, San Francisco, CA 94102. "
            "We're downtown near Union Square."
        )

    @function_tool
    async def confirm_appointment(self, context: RunContext, phone: str, time: str) -> str:
        """Confirm an appointment for the caller.

        Args:
            phone: The phone number to confirm the appointment for
            time: The requested appointment time
        """
        logger.info(f"Confirming appointment for {phone} at {time}")
        return f"I've confirmed your appointment for {phone} at {time}. You should receive a text reminder before your appointment."


server = AgentServer()


@server.rtc_session(agent_name="telnyx-agent")
async def entrypoint(ctx: JobContext):
    """Entry point for the Telnyx agent.

    This function is called when LiveKit dispatches an agent to handle
    an incoming call through the Telnyx SIP trunk.
    """
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "sip_trunk": SIP_TRUNK_ID[:8] + "..." if SIP_TRUNK_ID else "not configured",
        "sip_number": SIP_NUMBER,
    }

    session = AgentSession(
        # Speech-to-text (STT) - converts caller speech to text
        # Using Deepgram Nova-3 for telephony-optimized transcription
        stt=inference.STT("deepgram/nova-3", language="en"),
        # Large Language Model (LLM) - processes user input and generates responses
        # Using OpenAI GPT-4.1-mini for efficient, fast responses
        llm=inference.LLM("openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) - generates voice output for the caller
        # Using Cartesia Sonic for low-latency, natural speech
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        # Voice Activity Detection (VAD) - detects when the caller is speaking
        vad=ctx.proc.userdata.get("vad", silero.VAD.load()),
        # Turn detection - determines when to respond in the conversation
        turn_detection=MultilingualModel(),
        # Enable preemptive generation for faster response times
        preemptive_generation=True,
    )

    # Collect metrics for monitoring and debugging
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=TelnyxAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            audio_output=room_io.AudioOutputOptions(),
        ),
    )


def prewarm(proc):
    """Prewarm the job process with VAD model."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


if __name__ == "__main__":
    # Run the agent server
    # Use 'console' mode for testing without a frontend
    cli.run_app(server)
