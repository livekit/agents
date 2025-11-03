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

logger = logging.getLogger("elevenlabs-tts-per-reply")

load_dotenv()

# Test for mid-session settings updates with ElevenLabs TTS
class MyAgent(Agent):
    def __init__(self, tts: InferenceTTS) -> None:
        # Store TTS reference for mid-session updates
        self._tts = tts
        # Track reply count to cycle through different settings
        self._reply_count = 0
        # ElevenLabs voice IDs - you can get a list using 'await elevenlabs.TTS().list_voices()'
        self._voices = [
            "Xb7hH8MSUJpSbSDYk0k2",  # Default voice
            "nPczCjzI2devNBz1zQrb",  # Example voice 1
            "pFZP5JQG7iQjIQuC4Bku",  # Example voice 2 (Bella)
            "N2lVS1w4EtoT3dr4eOWO",
            "EXAVITQu4vr4xnSDxMaL",
            "SAz9YHcvj6GT2YYXdXww",
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
        # ElevenLabs supports: auto_mode (bool), apply_text_normalization ("on"/"off"/"auto"), inactivity_timeout (int)
        settings_cycle = [
            {"auto_mode": True, "apply_text_normalization": "on"},
            {"auto_mode": False, "apply_text_normalization": "off"},
            {"auto_mode": True, "apply_text_normalization": "auto"},
            {"auto_mode": False, "apply_text_normalization": "on"},
            {"auto_mode": True, "apply_text_normalization": "off", "inactivity_timeout": 120},
            {"auto_mode": False, "apply_text_normalization": "auto", "inactivity_timeout": 90},
        ]

        # Get settings for this reply (cycle through the list)
        settings = settings_cycle[(self._reply_count - 1) % len(settings_cycle)]
        # Get voice for this reply (cycle through the voices)
        voice_id = self._voices[(self._reply_count - 1) % len(self._voices)]

        # Update TTS settings before generating this reply
        logger.info(
            f"Reply #{self._reply_count}: Updating TTS voice={voice_id}, settings={settings}"
        )
        # Voice updates work via update_options
        self._tts.update_options(voice=voice_id)
        # Update extra_kwargs - these will be sent in input_transcript messages
        # to the gateway, enabling mid-session updates
        self._tts._opts.extra_kwargs.update(settings)

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

    # Create TTS instance with ElevenLabs model
    # You can specify a voice ID, or let it use the default
    tts = InferenceTTS(
        model="elevenlabs/eleven_flash_v2_5",
        voice="bIHbv24MWmeRgasZH58o",  # Default voice ID, adjust as needed
        # Optional: Set initial extra_kwargs
        extra_kwargs={"auto_mode": True, "apply_text_normalization": "auto"},
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
    #   uv run examples/voice_agents/elevenlabs_tts_per_reply.py download-files
    #
    # Then run the agent:
    #   uv run examples/voice_agents/elevenlabs_tts_per_reply.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
