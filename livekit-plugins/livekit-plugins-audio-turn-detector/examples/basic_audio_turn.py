"""Basic example of using AudioTurnDetector with LiveKit Agents.

This example demonstrates how to use audio-based turn detection
instead of or in addition to text-based turn detection.
"""

import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.audio_turn_detector import AudioTurnDetector

logger = logging.getLogger("audio-turn-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    # Load audio turn detector from the prewarm data
    audio_turn_detector = ctx.proc.userdata["audio_turn_detector"]

    # Create agent session with audio-based turn detection
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),  # Still need STT for transcription
        turn_detection=audio_turn_detector,  # Audio-based turn detector (unified!)
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=2.0,
    )

    await session.start(
        agent=Agent(
            instructions="You are a helpful voice assistant. Keep your responses concise."
        ),
        room=ctx.room,
    )


def prewarm(proc: JobProcess):
    """Prewarm function to load models before handling requests."""

    logger.info("Prewarming VAD and Audio Turn Detector...")

    # Load VAD
    proc.userdata["vad"] = silero.VAD.load()

    # Load Audio Turn Detector
    # Replace with your actual model path
    audio_turn_detector = AudioTurnDetector(
        model_path="path/to/your/audio_turn_model.onnx",
        feature_type="whisper",  # or "raw" depending on your model
        max_audio_seconds=8,
        activation_threshold=0.7,  # Adjust based on your model
        unlikely_threshold=0.3,  # Low probability = use longer delay
        cpu_count=2,
    )

    proc.userdata["audio_turn_detector"] = audio_turn_detector

    logger.info("Prewarm completed")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
