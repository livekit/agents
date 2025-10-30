import logging
from collections.abc import AsyncIterable
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.inference import TTS as InferenceTTS
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("cartesia-tts-per-reply")

load_dotenv()

# Test for https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion
class MyAgent(Agent):
    def __init__(self, tts: InferenceTTS) -> None:
        # Store TTS reference for mid-session updates
        self._tts = tts
        # Track reply count to cycle through different settings
        self._reply_count = 0
        self._voices = [
            "0834f3df-e650-4766-a20c-5a93a43aa6e3",  # Leo
            "6776173b-fd72-460d-89b3-d85812ee518d",  # Jace
            "c961b81c-a935-4c17-bfb3-ba2239de8c2f",  # Kyle
            "f4a3a8e4-694c-4c45-9ca0-27caf97901b5",  # Gavin
            "cbaf8084-f009-4838-a096-07ee2e6612b1",  # Maya
            "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",  # Tessa
            "cc00e582-ed66-4004-8336-0175b85c85f6",  # Dana
            "26403c37-80c1-4a1a-8692-540551ca2ae5",  # Marian
        ]
        super().__init__(
            instructions=(
                "Your name is Echo. You are a helpful voice assistant. "
                "Keep responses concise and to the point."
            ),
            tts=tts,
        )

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[bytes]:
        """Override TTS node to update settings before each reply"""
        self._reply_count += 1

        # Cycle through different TTS settings configurations
        settings_cycle = [
            {"speed": "normal", "emotion": "neutral", "volume": 1.0},
            {"speed": "normal", "emotion": "angry", "volume": 1.0},
            {"speed": "normal", "emotion": "excited", "volume": 1.0},
            {"speed": "normal", "emotion": "content", "volume": 1.0},
            {"speed": "normal", "emotion": "sad", "volume": 1.0},
            {"speed": "normal", "emotion": "scared", "volume": 1.0},
        ]

        # Get settings for this reply (cycle through the list)
        settings = settings_cycle[(self._reply_count - 1) % len(settings_cycle)]
        # Get voice for this reply (cycle through the voices)
        voice_id = self._voices[(self._reply_count - 1) % len(self._voices)]

        # Update TTS settings before generating this reply
        logger.info(
            f"Reply #{self._reply_count}: Updating TTS voice={voice_id}, settings={settings}"
        )
        self._tts.update_options(voice=voice_id, extra_kwargs=settings)

        # Call the default TTS node to generate audio
        return Agent.default.tts_node(self, text, model_settings)


def prewarm(proc: JobProcess):
    """Preload VAD model for faster startup"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create TTS instance with Cartesia sonic-3 model
    # You can specify a voice ID, or let it use the default
    tts = InferenceTTS(
        model="cartesia/sonic-3",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Example voice ID, adjust as needed
        # Optional: Set initial extra_kwargs
        extra_kwargs={"speed": "normal", "volume": 1.0},
    )

    # Create session with TTS instance
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4o-mini",
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=MyAgent(tts=tts),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    # Before running the agent, download required model files:
    #   uv run examples/voice_agents/cartesia_tts_per_reply.py download-files
    #
    # Then run the agent:
    #   uv run examples/voice_agents/cartesia_tts_per_reply.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

