"""
Example voice agent using Baseten STT, LLM, and TTS.

This demonstrates how to use Baseten plugins in a full LiveKit voice agent
with streaming TTS support.

Usage:
    # Set environment variables
    export LIVEKIT_URL="wss://your-livekit-server"
    export LIVEKIT_API_KEY="your-livekit-api-key"
    export LIVEKIT_API_SECRET="your-livekit-api-secret"
    export BASETEN_API_KEY="your-baseten-api-key"

    # Run in console mode (local testing without LiveKit server)
    python examples/other/baseten_voice_agent.py console

    # Run in dev mode (connects to LiveKit)
    python examples/other/baseten_voice_agent.py dev
"""

import os

from livekit import agents
from livekit.agents import Agent, AgentSession, MetricsCollectedEvent, RoomInputOptions, metrics
from livekit.plugins import baseten, openai, silero

try:
    from livekit.plugins import noise_cancellation

    HAS_NOISE_CANCELLATION = True
except ImportError:
    HAS_NOISE_CANCELLATION = False

# Disable turn detector for now due to speaking rate bug with stereo audio
HAS_TURN_DETECTOR = False

whisper_model_id = ""  # Add your whisper model id here
orpheus_model_id = ""  # Add your Orpheus model id here

BASETEN_API_KEY = os.getenv("BASETEN_API_KEY")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=baseten.STT(
            api_key=BASETEN_API_KEY,
            model_endpoint="wss://chain-jwd7ggwk.api.baseten.co/development/websocket",
        ),
        llm=openai.LLM(
            api_key=BASETEN_API_KEY,
            base_url="https://inference.baseten.co/v1",
            model="openai/gpt-oss-120b",
            tool_choice="auto",
        ),
        tts=baseten.TTS(
            api_key=BASETEN_API_KEY,
            # The TTS plugin will automatically convert this to websocket for streaming
            model_endpoint="wss://model-qklvjme3.api.baseten.co/environments/production/websocket",
        ),
        vad=silero.VAD.load(),
        turn_detection=None,  # Disabled - using VAD-only turn detection
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)  # logs to stdout
        usage_collector.collect(ev.metrics)  # accumulate stats

    async def log_usage():
        summary = usage_collector.get_summary()
        print(f"[📊 USAGE SUMMARY] {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Configure room input options
    room_input_opts = RoomInputOptions()
    if HAS_NOISE_CANCELLATION:
        # LiveKit Cloud enhanced noise cancellation
        # - If self-hosting, omit this parameter
        # - For telephony applications, use `BVCTelephony` for best results
        room_input_opts = RoomInputOptions(noise_cancellation=noise_cancellation.BVC())

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=room_input_opts,
    )

    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
