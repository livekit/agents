"""LiveKit Voice Agent with Hamming Telemetry

Send OTel traces, logs, and metrics from your voice agent to Hamming
for observability and evaluation.

Setup:
    pip install "livekit-agents[codecs]" livekit-plugins-hamming \
        livekit-plugins-openai livekit-plugins-silero

Environment variables:
    HAMMING_API_KEY      Your Hamming workspace API key
    LIVEKIT_URL          Your LiveKit server URL
    LIVEKIT_API_KEY      Your LiveKit API key
    LIVEKIT_API_SECRET   Your LiveKit API secret
"""

import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import hamming, openai, silero

logger = logging.getLogger("hamming-trace-example")

load_dotenv()

# Initialize Hamming telemetry once at module level.
# setup_hamming() is a singleton — only the first call configures providers.
# api_key defaults to HAMMING_API_KEY env var.
telemetry = hamming.setup_hamming()


@function_tool
async def lookup_weather(context: RunContext, location: str) -> str:
    """Called when the user asks for weather related information.

    Args:
        location: The location they are asking for
    """

    logger.info("Looking up weather for %s", location)

    return "sunny with a temperature of 70 degrees."


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant.",
            llm=openai.LLM(model="gpt-4o-mini"),
            stt=openai.STT(),
            tts=openai.TTS(voice="alloy"),
            tools=[lookup_weather],
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Flush telemetry on shutdown (use to_thread to avoid blocking the event loop)
    async def flush():
        await asyncio.to_thread(telemetry.force_flush)

    ctx.add_shutdown_callback(flush)

    # Start the voice agent
    session = AgentSession(vad=silero.VAD.load())

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
