import logging
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.inference import TTS as InferenceTTS
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("cartesia-tts-updates")

load_dotenv()

# Test for https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion
class MyAgent(Agent):
    def __init__(self, tts: InferenceTTS) -> None:
        # Store TTS reference for mid-session updates
        self._tts = tts
        super().__init__(
            instructions=(
                "Your name is Echo. You are a helpful voice assistant that can change "
                "your voice characteristics based on user requests. "
                "Keep responses concise and to the point. "
                "You can adjust your speaking speed, emotion, and volume when asked. "
                "If a user asks you to speak faster, slower, sound happier, or change your voice in any way, "
                "you should use the update_voice_settings function to make those changes."
            ),
            tts=tts,
        )

    @function_tool
    async def update_voice_settings(
        self,
        context: RunContext,
        speed: str | None = None,
        emotion: str | None = None,
        volume: float | None = None,
        speed_config: float | None = None,
    ) -> str:
        """Update the voice settings for the assistant.
        
        This function allows you to change how the assistant sounds when speaking.
        Call this function when the user requests voice changes like speaking faster, 
        slower, or with different emotions.

        Args:
            speed: Speaking speed - one of "slow", "normal", "fast"
            emotion: Voice emotion (e.g., "happy", "sad", "excited", "calm")
            volume: Volume level between 0.6 and 1.5 (default 1.0)
            speed_config: Fine-grained speed control between 0.5 and 2.0 (overrides speed if provided)
        """
        extra_kwargs: dict[str, Any] = {}

        if speed_config is not None:
            if not 0.5 <= speed_config <= 2.0:
                return f"Speed config must be between 0.5 and 2.0, got {speed_config}"
            extra_kwargs["speed_config"] = speed_config
            logger.info(f"Updating speed_config to {speed_config}")
        elif speed is not None:
            if speed not in ["slow", "normal", "fast"]:
                return f"Speed must be one of 'slow', 'normal', 'fast', got '{speed}'"
            extra_kwargs["speed"] = speed
            logger.info(f"Updating speed to {speed}")

        if emotion is not None:
            extra_kwargs["emotion"] = emotion
            logger.info(f"Updating emotion to {emotion}")

        if volume is not None:
            if not 0.6 <= volume <= 1.5:
                return f"Volume must be between 0.6 and 1.5, got {volume}"
            extra_kwargs["volume"] = volume
            logger.info(f"Updating volume to {volume}")

        if extra_kwargs:
            self._tts.update_options(extra_kwargs=extra_kwargs)
            settings_desc = ", ".join(f"{k}={v}" for k, v in extra_kwargs.items())
            return f"Voice settings updated: {settings_desc}"
        else:
            return "No valid settings provided. Please specify speed, emotion, volume, or speed_config."

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up weather information for a given location.

        Args:
            location: The city or location to get weather for
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 72 degrees Fahrenheit."


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
        voice="0834f3df-e650-4766-a20c-5a93a43aa6e3",  # Example voice ID, adjust as needed
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
    #   uv run examples/voice_agents/cartesia_tts_updates.py download-files
    #
    # Then run the agent:
    #   uv run examples/voice_agents/cartesia_tts_updates.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

